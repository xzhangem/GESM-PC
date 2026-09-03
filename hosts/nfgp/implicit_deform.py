import os
import torch
import importlib
import os.path as osp
from argparse import Namespace
import torch.nn.functional as F
from trainers.base_trainer import BaseTrainer
from trainers.utils.utils import set_random_seed
from trainers.utils.igp_utils import sample_points
from trainers.losses.eikonal_loss import loss_eikonal
from models.igp_wrapper import distillation, deformation
# GESM-PC replaces the original Implicit Thin Shell (stretch + bending) losses.
# See trainers/losses/gesm_pc_losses.py and GESM-PC paper Eq. (14).
from trainers.losses.gesm_pc_losses import gesm_pc_loss
# Keep original imports available for ablation / fallback:
from trainers.losses.implicit_thin_shell_losses import \
    stretch_loss as stretch_loss_orig, bending_loss as bending_loss_orig


def deform_step(
        net, opt, original, handles_ts, targets_ts, dim=3,
        # Clip gradient
        grad_clip=None,
        # Sample points
        sample_cfg=None, x=None, weights=1,
        # Loss handle
        loss_h_weight=1., use_l1_loss=False, loss_h_thr=None,
        # Loss G
        loss_g_weight=1e-2, n_g_pts=5000,
        # ---- GESM-PC / legacy stretch+bend interface ----
        # loss_stretch_weight maps to GESM shear (a1); scale (b1) defaults to 0
        # loss_hess_weight    maps to GESM bend  (c1)
        loss_hess_weight=0., n_hess_pts=5000, hess_use_surf_points=True,
        hess_invert_sample=True, hess_detach_weight=True, hess_use_rejection=False,
        loss_stretch_weight=0., n_s_pts=5000, stretch_use_surf_points=True,
        stretch_invert_sample=True, stretch_loss_type='area_length',
        stretch_use_weight=False, stretch_detach_weight=True,
        stretch_use_rejection=False,
        # Explicit GESM-PC component weights (override mapping when not None)
        gesm_weight_shear=None, gesm_weight_scale=None,
        gesm_weight_bend=None, gesm_weight_smooth=0.,
        gesm_weight_jtj=0.,
        # Set True to fall back to the original NFGP thin-shell losses
        use_original_thin_shell=False,
):
    opt.zero_grad()

    # Compute handle losses
    handles_ts = handles_ts.clone().detach().float().cuda()
    targets_ts = targets_ts.clone().detach().float().cuda()
    constr = (
            net(targets_ts, return_delta=True)[0] + targets_ts - handles_ts
    ).view(-1, dim).norm(dim=-1, keepdim=False)
    if loss_h_thr is not None:
        loss_h_thr = float(loss_h_thr)
        constr = F.relu(constr - loss_h_thr)
    if use_l1_loss:
        loss_h = F.l1_loss(
            constr, torch.zeros_like(constr)) * loss_h_weight
    else:
        loss_h = F.mse_loss(
            constr, torch.zeros_like(constr)) * loss_h_weight

    if sample_cfg is not None and x is None:
        x, weights = sample_points(
            npoints=getattr(sample_cfg, "num_points", 5000),
            dim=dim, inp_nf=original, out_nf=net, deform=net.deform,
            sample_surf_points=getattr(sample_cfg, "use_surf_points", True),
            invert_sampling=getattr(sample_cfg, "invert_sample", True),
            detach_weight=getattr(sample_cfg, "detach_weight", True),
            use_rejection=getattr(sample_cfg, "use_rejection", False)
        )

    if loss_g_weight > 0.:
        try:
            loss_g = loss_eikonal(net, npoints=n_g_pts, dim=dim) * loss_g_weight
        except TypeError:
            # some versions accept an extra x= argument
            loss_g = loss_eikonal(net, npoints=n_g_pts, dim=dim, x=x) * loss_g_weight
    else:
        loss_g = torch.zeros(1).cuda().float()

    # ----------------------------------------------------------------
    # Elastic regularisation
    # ----------------------------------------------------------------
    n_elastic_pts = max(int(n_hess_pts), int(n_s_pts))
    use_surf = hess_use_surf_points or stretch_use_surf_points
    invert_smp = hess_invert_sample or stretch_invert_sample
    detach_w = hess_detach_weight and stretch_detach_weight
    use_rej = hess_use_rejection or stretch_use_rejection

    zero = torch.zeros(1).cuda().float()

    if use_original_thin_shell:
        # ---- original NFGP Implicit Thin Shell (ablation) ----
        if loss_hess_weight > 0.:
            loss_hess = bending_loss_orig(
                inp_nf=original, out_nf=net, deform=net.deform,
                dim=dim, npoints=n_hess_pts,
                use_surf_points=hess_use_surf_points,
                invert_sampling=hess_invert_sample,
                x=x, weights=weights,
                detach_weight=hess_detach_weight,
                use_rejection=hess_use_rejection,
            ) * loss_hess_weight
        else:
            loss_hess = zero
        if loss_stretch_weight > 0.:
            loss_stretch = stretch_loss_orig(
                inp_nf=original, out_nf=net, deform=net.deform,
                npoints=n_s_pts, dim=dim,
                use_surf_points=stretch_use_surf_points,
                invert_sampling=stretch_invert_sample,
                loss_type=stretch_loss_type,
                x=x, weights=weights,
                detach_weight=stretch_detach_weight,
                use_rejection=stretch_use_rejection,
            ) * loss_stretch_weight
        else:
            loss_stretch = zero
        loss_gesm_shear = zero
        loss_gesm_scale = zero
        loss_gesm_jtj = zero
        loss_gesm_bend = zero
        loss_gesm_smooth = zero
        loss = loss_h + loss_g + loss_hess + loss_stretch
    else:
        # ---- GESM-PC (Eq. 14) ----
        w_shear = float(gesm_weight_shear) if gesm_weight_shear is not None \
            else float(loss_stretch_weight)
        # Scaling term off by default (do not inherit stretch weight).
        w_scale = float(gesm_weight_scale) if gesm_weight_scale is not None \
            else 0.0
        w_bend = float(gesm_weight_bend) if gesm_weight_bend is not None \
            else float(loss_hess_weight)
        w_smooth = float(gesm_weight_smooth)
        w_jtj = float(gesm_weight_jtj)

        if (w_shear + w_scale + w_bend + w_smooth + w_jtj) > 0.:
            loss_gesm, comps = gesm_pc_loss(
                inp_nf=original, out_nf=net, deform=net.deform,
                x=x, weights=weights,
                npoints=n_elastic_pts, dim=dim,
                use_surf_points=use_surf,
                invert_sampling=invert_smp,
                detach_weight=detach_w,
                use_rejection=use_rej,
                weight_shear=w_shear,
                weight_scale=w_scale,
                weight_bend=w_bend,
                weight_smooth=w_smooth,
                weight_jtj=w_jtj,
                return_components=True,
            )
            loss_gesm_shear = comps['loss_shear']
            loss_gesm_scale = comps['loss_scale']
            loss_gesm_jtj = comps.get('loss_jtj', zero)
            loss_gesm_bend = comps['loss_bend']
            loss_gesm_smooth = comps['loss_smooth']
            # stretch group = shear + scale + ||L^T L||_F
            loss_stretch = comps.get(
                'loss_stretch',
                loss_gesm_shear + loss_gesm_scale + loss_gesm_jtj)
            loss_hess = loss_gesm_bend
        else:
            loss_gesm = zero
            loss_gesm_shear = zero
            loss_gesm_scale = zero
            loss_gesm_jtj = zero
            loss_gesm_bend = zero
            loss_gesm_smooth = zero
            loss_stretch = zero
            loss_hess = zero
        loss = loss_h + loss_g + loss_gesm

    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.deform.parameters(), grad_clip)

    opt.step()

    def _item(t):
        return t.detach().cpu().item() if torch.is_tensor(t) else float(t)

    return {
        'loss': _item(loss),
        'loss_h': _item(loss_h),
        'loss_g': _item(loss_g),
        'loss_hess': _item(loss_hess),
        'loss_stretch': _item(loss_stretch),
        'loss_gesm_shear': _item(loss_gesm_shear),
        'loss_gesm_scale': _item(loss_gesm_scale),
        'loss_gesm_jtj': _item(loss_gesm_jtj),
        'loss_gesm_bend': _item(loss_gesm_bend),
        'loss_gesm_smooth': _item(loss_gesm_smooth),
    }



class Trainer(BaseTrainer):

    def __init__(self, cfg, args, original_decoder=None):
        super().__init__(cfg, args)
        self.cfg = cfg
        self.args = args
        set_random_seed(getattr(self.cfg.trainer, "seed", 666))

        # The networks
        # TODO: add recursive loading of trainers.
        if original_decoder is None:
            sn_lib = importlib.import_module(cfg.models.decoder.type)
            self.original_net = sn_lib.Net(cfg, cfg.models.decoder)
            self.original_net.cuda()
            self.original_net.load_state_dict(
                torch.load(cfg.models.decoder.path)['net'])
            print("Original Decoder:")
            print(self.original_net)
        else:
            self.original_net = original_decoder

        # Get the wrapper for the operation
        self.wrapper_type = getattr(
            cfg.trainer, "wrapper_type", "distillation")
        if self.wrapper_type in ['distillation']:
            self.net, self.opt, self.sch = distillation(
                cfg, self.original_net,
                reload=getattr(self.cfg.trainer, "reload_decoder", True))
        elif self.wrapper_type in ['deformation']:
            self.net, self.opt, self.sch = deformation(
                cfg, self.original_net)
        else:
            raise ValueError("wrapper_type:", self.wrapper_type)

        # Prepare save directory
        os.makedirs(osp.join(cfg.save_dir, "images"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "val"), exist_ok=True)
        os.makedirs(osp.join(cfg.save_dir, "vis"), exist_ok=True)

        # Set-up counter
        self.num_update_step = 0
        self.boundary_points = None

        # Set up basic parameters
        self.dim = getattr(cfg.trainer, "dim", 3)
        self.grad_clip = getattr(cfg.trainer, "grad_clip", None)
        self.loss_h_weight = getattr(cfg.trainer, "loss_h_weight", 100)
        self.loss_h_thr = getattr(cfg.trainer, "loss_h_thr", 1e-3)

        if hasattr(cfg.trainer, "loss_g"):
            self.loss_g_cfg = cfg.trainer.loss_g
        else:
            self.loss_g_cfg = Namespace(**{})

        if hasattr(cfg.trainer, "loss_bend"):
            self.loss_bend_cfg = cfg.trainer.loss_bend
        else:
            self.loss_bend_cfg = Namespace(**{})

        if hasattr(cfg.trainer, "loss_stretch"):
            self.loss_stretch_cfg = cfg.trainer.loss_stretch
        else:
            self.loss_stretch_cfg = Namespace()

        # Optional explicit GESM-PC config block:
        #   loss_gesm:
        #     weight_shear:  ...
        #     weight_scale:  ...
        #     weight_bend:   ...
        #     weight_smooth: ...
        #     use_original_thin_shell: false
        if hasattr(cfg.trainer, "loss_gesm"):
            self.loss_gesm_cfg = cfg.trainer.loss_gesm
        else:
            self.loss_gesm_cfg = Namespace()

        if hasattr(cfg.trainer, "sample_cfg"):
            self.sample_cfg = cfg.trainer.sample_cfg
        else:
            self.sample_cfg = None

        self.show_network_hist = getattr(
            cfg.trainer, "show_network_hist", False)

    def update(self, data, *args, **kwargs):
        self.num_update_step += 1
        handles_ts = data['handles'].cuda().float()
        targets_ts = data['targets'].cuda().float()
        if 'x' in data and 'weights' in data:
            x_ts = data['x'].cuda().float()
            w_ts = data['weights'].cuda().float()
        else:
            x_ts = None
            w_ts = 1.

        loss_g_weight = float(getattr(self.loss_g_cfg, "weight", 1e-3))
        loss_hess_weight = float(getattr(self.loss_bend_cfg, "weight", 0.))
        loss_stretch_weight = float(
            getattr(self.loss_stretch_cfg, "weight", 0))

        # Explicit GESM-PC weights (None -> fall back to stretch/bend mapping)
        def _gesm_w(name):
            v = getattr(self.loss_gesm_cfg, name, None)
            return None if v is None else float(v)

        step_res = deform_step(
            self.net, self.opt, self.original_net,
            handles_ts, targets_ts, dim=self.dim,
            x=x_ts, weights=w_ts,
            sample_cfg=self.sample_cfg,
            # Loss handle
            loss_h_weight=self.loss_h_weight,
            loss_h_thr=self.loss_h_thr,
            # Loss G
            loss_g_weight=loss_g_weight,
            n_g_pts=getattr(self.loss_g_cfg, "num_points", 5000),

            # Legacy stretch / bend (used as default mapping into GESM)
            loss_hess_weight=loss_hess_weight,
            n_hess_pts=getattr(self.loss_bend_cfg, "num_points", 5000),
            hess_use_surf_points=getattr(
                self.loss_bend_cfg, "use_surf_points", True),
            hess_invert_sample=getattr(
                self.loss_bend_cfg, "invert_sample", True),
            hess_detach_weight=getattr(
                self.loss_bend_cfg, "detach_weight", True),
            hess_use_rejection=getattr(
                self.loss_bend_cfg, "use_rejection", True),
            loss_stretch_weight=loss_stretch_weight,
            n_s_pts=getattr(self.loss_stretch_cfg, "num_points", 5000),
            stretch_use_surf_points=getattr(
                self.loss_stretch_cfg, "use_surf_points", True),
            stretch_invert_sample=getattr(
                self.loss_stretch_cfg, "invert_sample", True),
            stretch_loss_type=getattr(
                self.loss_stretch_cfg, "loss_type", "l2"),
            stretch_use_weight=getattr(
                self.loss_stretch_cfg, "use_weight", True),
            stretch_detach_weight=getattr(
                self.loss_stretch_cfg, "detach_weight", True),
            stretch_use_rejection=getattr(
                self.loss_stretch_cfg, "use_rejection", True),

            # Explicit GESM-PC component weights
            gesm_weight_shear=_gesm_w("weight_shear"),
            gesm_weight_scale=_gesm_w("weight_scale"),
            gesm_weight_bend=_gesm_w("weight_bend"),
            gesm_weight_smooth=float(
                getattr(self.loss_gesm_cfg, "weight_smooth", 0.)),
            gesm_weight_jtj=float(
                getattr(self.loss_gesm_cfg, "weight_jtj", 0.)),
            use_original_thin_shell=bool(
                getattr(self.loss_gesm_cfg, "use_original_thin_shell", False)),

            # Gradient clipping
            grad_clip=self.grad_clip,
        )
        # Keep plain names for terminal logging; scalar/* for TensorBoard
        plain = dict(step_res)
        step_res = {
            ('scalar/loss/%s' % k): v for k, v in plain.items()
        }
        step_res.update(plain)  # loss, loss_h, loss_stretch, loss_hess, ...
        step_res['loss'] = plain['loss']
        step_res.update({
            "scalar/weight/loss_h_weight": self.loss_h_weight,
            'scalar/weight/loss_hess_weight': loss_hess_weight,
            'scalar/weight/loss_stretch_weight': loss_stretch_weight,
        })
        return step_res

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False, **kwargs):
        if writer is None:
            return
        writer_step = step if step is not None else epoch

        # Log training information to tensorboard
        train_info = {k: (v.cpu() if not isinstance(v, float) else v)
                      for k, v in train_info.items()}
        for k, v in train_info.items():
            ktype = k.split("/")[0]
            kstr = "/".join(k.split("/")[1:])
            if ktype == 'scalar':
                writer.add_scalar(kstr, v, writer_step)

        if self.show_network_hist:
            for name, p in self.net.named_parameters():
                writer.add_histogram("dec/%s" % name, p, writer_step)
            for name, p in self.original_net.named_parameters():
                writer.add_histogram("orig_dec/%s" % name, p, writer_step)

    def validate(self, test_loader, epoch, *args, **kwargs):
        # TODO: compute mesh and compute the manifold harmonics to
        #       see if the high frequencies signals are dimed/suppressed
        return {}

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        d = {
            'dec': self.original_net.state_dict(),
            'net_opt_dec': self.opt.state_dict(),
            'next_dec': self.net.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        torch.save(d, osp.join(self.cfg.save_dir, "checkpoints", save_name))
        torch.save(d, osp.join(self.cfg.save_dir, "latest.pt"))

    def resume(self, path, strict=True, **kwargs):
        ckpt = torch.load(path)
        self.original_net.load_state_dict(ckpt['dec'], strict=strict)
        self.net.load_state_dict(ckpt['next_dec'], strict=strict)
        self.opt.load_state_dict(ckpt['net_opt_dec'])
        start_epoch = ckpt['epoch']
        return start_epoch

    def multi_gpu_wrapper(self, wrapper):
        self.net = wrapper(self.net)

    def epoch_end(self, epoch, writer=None, **kwargs):
        if self.sch is not None:
            self.sch.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'lr/opt_dec_lr_sch', self.sch.get_lr()[0], epoch)

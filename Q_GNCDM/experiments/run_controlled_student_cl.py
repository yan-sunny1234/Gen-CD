"""Controlled Math1 same-student continuous-noise InfoNCE experiment."""
from Q_GNCDM.experiments.run_simple_contrastive import *
from Q_GNCDM.experiments import run_simple_contrastive as core
from Q_GNCDM.experiments import reference_student_cl_acc as reference
from Q_GNCDM.experiments.controlled_student_cl import (
    augment, same_student_infonce, contrastive_branch, shared_parameters,
    gradient_diagnostics,
)
import traceback
from unittest.mock import patch

def controlled_train_one(data, args, mode, seed, output):
    output.mkdir(parents=True, exist_ok=False)
    settings = MODES[mode]
    model = build_model(data, args, mode, seed)
    optimizer = build_optimizer(model, args)
    eligible = np.flatnonzero((data["train"] != 0).sum(axis=1) >= 2)
    if len(eligible) < 2:
        raise ValueError("At least two eligible students are required")
    legacy.write_json(output / "config.json", {
        "args": vars(args),
        "mode": mode,
        "settings": settings,
        "seed": seed,
        "source_sha256": data["source_sha256"],
        "protocol": (
            "seen-student; clean support for prediction; continuous observed-only support "
            "for same-student InfoNCE; query hidden from student and item inputs"
        ),
    })

    best_state = deepcopy(model.state_dict())
    best_auc, best_epoch, bad_epochs = -float("inf"), 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng([seed, epoch, 0]).permutation(eligible)
        epoch_rows = []
        for step, start in enumerate(range(0, len(order), args.batch_students)):
            students = order[start:start + args.batch_students]
            if len(students) < 2:
                continue
            episode = make_student_episode(
                data["train"], students, args.query_fraction,
                np.random.default_rng([seed, epoch, step, 1]),
            )
            clean_np = episode.support[students]
            clean = torch.as_tensor(clean_np, dtype=torch.float32, device=model.device)
            theta_clean = model.diagnose_theta(clean)

            item_ids, inverse = np.unique(episode.query_items, return_inverse=True)
            item_logs = torch.as_tensor(
                episode.support[:, item_ids].T.copy(),
                dtype=torch.float32,
                device=model.device,
            )
            psi = model.diagnose_psi(item_logs)[
                torch.as_tensor(inverse, device=model.device)
            ]
            query_students = torch.as_tensor(
                episode.query_students, device=model.device
            )
            query_items = torch.as_tensor(episode.query_items, device=model.device)
            prediction = model.predict_response(
                theta_clean[query_students],
                psi,
                model.get_q_soft()[query_items],
            )
            target = torch.as_tensor(
                episode.query_labels, dtype=torch.float32, device=model.device
            ).reshape(-1, 1)
            losses = model.compute_loss(prediction, target)

            diagnostics = {}
            if args.lambda_cl > 0:
                loss_cl, diagnostics = contrastive_branch(
                    model, clean_np, theta_clean, args, seed, epoch, step)
                grad_stats, _ = gradient_diagnostics(
                    losses["loss_total"], loss_cl, model, args.lambda_cl)
                diagnostics.update(grad_stats)
                total = losses["loss_total"] + args.lambda_cl * loss_cl
                coefficient = args.lambda_cl
            else:
                # No RNG creation, no augmentation, no extra forward/autograd.
                loss_cl = None
                coefficient = 0.0
                total = losses["loss_total"]
            flip_fraction = 0.0
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            epoch_rows.append({
                "loss_total": float(total.detach().cpu()),
                "loss_pred": float(losses["loss_rec"].detach().cpu()),
                "loss_q_regularization": float(
                    (losses["loss_total"] - losses["loss_rec"]).detach().cpu()
                ),
                "loss_cl": 0.0 if loss_cl is None else float(loss_cl.detach().cpu()),
                "loss_original": float(losses["loss_total"].detach().cpu()),
                "weighted_loss_cl": 0.0 if loss_cl is None else coefficient * float(loss_cl.detach().cpu()),
                **diagnostics,
                "lambda_cl_effective": coefficient,
                "actual_flip_fraction": flip_fraction,
            })

        valid, theta_valid, _ = evaluate(model, data["train"], data["rows"][1])
        record = {
            "epoch": epoch,
            **{key: float(np.mean([row[key] for row in epoch_rows]))
               for key in epoch_rows[0]},
            **{f"valid_{key}": value for key, value in valid.items()},
            "theta_std_mean": float(theta_valid.std(axis=0).mean()),
        }
        history.append(record)
        score = valid["acc"]  # Preregistered ACC selection; earliest tie.
        if score > best_auc:
            best_auc, best_epoch = score, epoch
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        save_csv(output / "train_log.csv", history)
        print(
            f"{args.dataset} {mode} seed={seed} epoch={epoch} "
            f"pred={record['loss_pred']:.4f} cl={record['loss_cl']:.4f} "
            f"valid_auc={valid['auc']}",
            flush=True,
        )
        if args.patience > 0 and bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "mode": mode,
        "seed": seed,
        "best_epoch": best_epoch,
    }, output / "best_model.pt")
    valid, _, _ = evaluate(model, data["train"], data["rows"][1])
    test, theta, probabilities = evaluate(model, data["train"], data["rows"][2])
    q_reference = (
        data["q"] if args.dataset == "math1"
        else np.load(data["folder"] / "Q_true.npy")
    )
    q_scores = legacy.q_recovery_metrics(
        q_reference,
        data["q"],
        model.get_q_hard().cpu().numpy(),
        model.get_q_soft().detach().cpu().numpy(),
    )
    result = {
        "dataset": args.dataset,
        "mode": mode,
        "mode_label": settings["label"],
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "seconds": time.perf_counter() - started,
        **{f"valid_{key}": value for key, value in valid.items()},
        **{f"test_{key}": value for key, value in test.items()},
        "theta_std_mean": float(theta.std(axis=0).mean()),
        "q_f1_vs_reference": q_scores["q_hat_f1_vs_true"],
        "mean_actual_flip_fraction": float(
            np.mean([row["actual_flip_fraction"] for row in history])
        ),
    }
    result["training_means"] = {key: float(np.mean([r[key] for r in history]))
                                for key in history[0] if not key.startswith("valid_")}
    result["margin_activation"] = None  # InfoNCE has no margin.
    if args.dataset == "synthetic":
        mastery_true = np.load(data["folder"] / "mastery_true.npy")
        result.update(legacy.mastery_metrics(theta, mastery_true))
    legacy.write_json(output / "metrics.json", result)
    return result


def batch_forward(model, episode):
    """Exactly the original grouped clean prediction path, for isolated tests."""
    clean_np = episode.support[episode.student_ids]
    clean = torch.as_tensor(clean_np, dtype=torch.float32, device=model.device)
    theta = model.diagnose_theta(clean)
    item_ids, inverse = np.unique(episode.query_items, return_inverse=True)
    item_logs = torch.as_tensor(episode.support[:, item_ids].T.copy(),
                                dtype=torch.float32, device=model.device)
    psi = model.diagnose_psi(item_logs)[torch.as_tensor(inverse, device=model.device)]
    users = torch.as_tensor(episode.query_students, device=model.device)
    items = torch.as_tensor(episode.query_items, device=model.device)
    prediction = model.predict_response(theta[users], psi, model.get_q_soft()[items])
    target = torch.as_tensor(episode.query_labels, dtype=torch.float32,
                             device=model.device).reshape(-1, 1)
    losses = model.compute_loss(prediction, target)
    return clean_np, theta, prediction, losses


def preflight(data, args, mode):
    checks = {}
    x = np.tile(np.array([-1, 0, 1, 1, -1], dtype=np.float32), (20000, 1))
    before = x.copy()
    samples = []
    for tag, ratio, scale in (("positive", args.positive_noise_ratio, args.positive_noise_scale),
                               ("negative", args.negative_noise_ratio, args.negative_noise_scale)):
        y, st = augment(x, ratio, scale, np.random.default_rng(123))
        again, _ = augment(x, ratio, scale, np.random.default_rng(123))
        other, _ = augment(x, ratio, scale, np.random.default_rng(124))
        assert np.array_equal(y, again) and not np.array_equal(y, other)
        assert y.dtype == x.dtype and y.shape == x.shape and np.isfinite(y).all()
        assert np.array_equal(y[x == 0], x[x == 0]) and np.abs(y).max() <= 1
        assert np.array_equal(np.sign(y), np.sign(x))
        assert abs(st['selected_fraction'] - ratio) < .01
        assert st['input_mae_observed'] > 0
        for r, s in ((0, scale), (ratio, 0)):
            assert np.array_equal(augment(x, r, s, np.random.default_rng(1))[0], x)
        checks[tag] = st
        samples.append(st['input_mae_observed'])
    assert samples[0] < samples[1] and np.array_equal(x, before)
    model = build_model(data, args, mode, 2026)
    initial = deepcopy(model.state_dict())
    eligible = np.flatnonzero((data['train'] != 0).sum(1) >= 2)[:args.batch_students]
    ep = make_student_episode(data['train'], eligible, args.query_fraction,
                              np.random.default_rng([2026, 1, 0, 1]))
    assert np.array_equal(ep.student_ids, eligible)
    original_train = data['train'].copy()
    rng = torch.get_rng_state().clone()
    cases = []
    for coefficient in (0., .1):
        model.load_state_dict(initial)
        optimizer = build_optimizer(model, args)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        torch.set_rng_state(rng)
        with patch.object(model, 'diagnose_theta', wraps=model.diagnose_theta) as spy:
            clean, theta, prediction, losses = batch_forward(model, ep)
            rng_after_prediction = torch.get_rng_state().clone()
            if coefficient:
                cl, diag = contrastive_branch(model, clean, theta, args, 2026, 1, 0)
                assert cl.ndim == 0 and bool(torch.isfinite(cl)) and cl.requires_grad
                grad, per_layer = gradient_diagnostics(losses['loss_total'], cl, model, coefficient)
                assert grad['cl_shared_grad_norm'] > 0
                total = losses['loss_total'] + coefficient * cl
                assert torch.equal(total, losses['loss_total'] + .1 * cl)
                checks.update({'gradient': grad, 'per_layer': per_layer, 'batch_diagnostics': diag,
                               'loss_original': float(losses['loss_total'].detach()),
                               'loss_cl': float(cl.detach()), 'loss_total': float(total.detach())})
            else:
                total = losses['loss_total']
            assert spy.call_count == (3 if coefficient else 1)
            assert torch.equal(torch.get_rng_state(), rng_after_prediction)
            total.backward()
            gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            optimizer.step()
        cases.append((prediction.detach().clone(), gradients, deepcopy(model.state_dict())))
    assert torch.equal(cases[0][0], cases[1][0])
    assert np.array_equal(data['train'], original_train)
    changes = {}
    for name, _ in shared_parameters(model):
        g0, g1 = cases[0][1][name], cases[1][1][name]
        w0, w1 = cases[0][2][name], cases[1][2][name]
        changes[name] = {
            'gradient_difference_l2': float((g1 - g0).double().norm()),
            'update_difference_l2': float((w1 - w0).double().norm()),
            'baseline_update_l2': float((w0 - initial[name]).double().norm()),
            'cl_update_l2': float((w1 - initial[name]).double().norm()),
        }
    assert any(v['update_difference_l2'] > 0 for k, v in changes.items() if k.startswith('f_nn.'))
    assert any(v['gradient_difference_l2'] > 0 for v in changes.values())
    assert any(v['baseline_update_l2'] > 0 for v in changes.values())
    checks['single_step_parameter_changes'] = changes
    # Separate exact R0/R1 step parity; both use the unchanged model and optimizer.
    parity = []
    for reference_path in (True, False):
        model.load_state_dict(initial)
        optimizer = build_optimizer(model, args)
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        _, theta, pred, ls = batch_forward(model, ep)
        total = ls['loss_total'] + 0.0 * (theta.sum() * 0.0) if reference_path else ls['loss_total']
        total.backward()
        gs = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        parity.append((pred.detach().clone(), total.detach().clone(), gs, deepcopy(model.state_dict())))
    assert torch.equal(parity[0][0], parity[1][0]) and torch.equal(parity[0][1], parity[1][1])
    assert all(torch.equal(parity[0][2][k], parity[1][2][k]) for k in parity[0][2])
    assert all(torch.equal(parity[0][3][k], parity[1][3][k]) for k in parity[0][3])
    checks['zero_single_step_exact_parity'] = True
    checks['status'] = 'passed'
    return checks


def checkpoint_state(path):
    return torch.load(path / 'best_model.pt', map_location='cpu', weights_only=True)['model_state_dict']


def export_weights(path, state):
    arrays = {k: v.cpu().numpy() for k, v in state.items() if k.endswith(('.weight', '.bias')) or k == 'q_logits'}
    for k in list(arrays):
        if k.endswith('.weight') and (k.startswith('f_nn.') or k.startswith('theta_agg_mat.')):
            arrays[k + '.effective'] = np.abs(arrays[k])
    np.savez_compressed(path / 'weights.npz', **arrays)


def run_experiment(args):
    torch.set_num_threads(1)
    data = core.load_data(args)
    destination = ROOT / 'Q_GNCDM/results' / args.experiment_id
    destination.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), Path(reference.__file__),
                    ROOT / 'Q_GNCDM/experiments/controlled_student_cl.py',
                    Path(core.__file__), ROOT / 'Q_GNCDM/models/weak_q_gncdm.py',
                    ROOT / 'Q_GNCDM/simple_contrastive.py', Path(legacy.__file__)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    manifest = {'args': vars(args), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'git_status_start': subprocess.check_output(['git', 'status', '--short'], text=True),
                'code_sha256': hashes, 'data_sha256': data['source_sha256'],
                'split_lengths': [len(r) for r in data['rows']],
                'command': subprocess.list2cmdline([sys.executable, '-m', 'Q_GNCDM.experiments.run_controlled_student_cl', *sys.argv[1:]]),
                'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                'selection': 'maximum validation ACC; earliest tie; unchanged patience 8',
                'baseline': 'R0: original clean training loop, only ACC selector and artifact reduction adapted',
                'single_change_R1_R2': 'lambda_cl 0 versus 0.1',
                'noise': 'Bernoulli element selection, uniform additive noise, clip [-1,1]; student support only',
                'loss': '-log softmax([cos(m,mp),cos(m,mn)] / temperature)[0]; one same-student negative',
                'margin_activation': None, 'margin_note': 'not applicable to InfoNCE',
                'gradient_sampling': 'every R2 training batch; autograd.grad does not accumulate into .grad',
                'matrix_policy': 'all learned weights and biases in per-run weights.npz; full state in checkpoint'}
    legacy.write_json(destination / 'manifest.json', manifest)
    try:
        checks = {}
        for mode in ('A0', 'A1'):
            checks[mode] = preflight(data, args, mode)
            legacy.write_json(destination / 'preflight.json', checks)
        print('PREFLIGHT PASSED for fixed Q and Weak-Q', flush=True)
        records, comparisons, parity_results = [], [], []
        for seed in args.seeds:
            for mode in ('A0', 'A1'):
                paths = {}
                for condition, coefficient in (('R0', 0.), ('R1', 0.), ('R2', args.lambda_cl)):
                    cfg = deepcopy(args)
                    cfg.lambda_cl = coefficient
                    output = destination / mode / f'seed_{seed}' / condition
                    paths[condition] = output
                    if condition == 'R0':
                        result = reference.train_one(data, cfg, mode, seed, output)
                    elif condition == 'R1':
                        # Hard assertion: zero coefficient must never enter CL helper.
                        with patch(__name__ + '.contrastive_branch', side_effect=AssertionError('zero branch called CL')):
                            result = controlled_train_one(data, cfg, mode, seed, output)
                    else:
                        result = controlled_train_one(data, cfg, mode, seed, output)
                    result.update({'condition': condition, 'lambda_cl': coefficient})
                    legacy.write_json(output / 'metrics.json', result)
                    state = checkpoint_state(output)
                    export_weights(output, state)
                    flat = {k: v for k, v in result.items() if not isinstance(v, dict)}
                    flat.update(result.get('training_means', {}))
                    records.append(flat)
                    core.save_csv(destination / 'summary.csv', records)
                    if condition == 'R1':
                        baseline = checkpoint_state(paths['R0'])
                        assert all(torch.equal(state[k], baseline[k]) for k in state), 'R0/R1 checkpoint mismatch'
                        r0 = list(csv.DictReader((paths['R0'] / 'train_log.csv').open(encoding='utf-8-sig')))
                        r1 = list(csv.DictReader((paths['R1'] / 'train_log.csv').open(encoding='utf-8-sig')))
                        assert len(r0) == len(r1)
                        for a, b in zip(r0, r1):
                            assert all(a[k] == b[k] for k in a), 'R0/R1 epoch history mismatch'
                        m0 = json.loads((paths['R0'] / 'metrics.json').read_text(encoding='utf-8'))
                        assert all(m0[k] == result[k] for k in m0 if k.startswith(('test_', 'valid_')))
                        parity_results.append({'mode': mode, 'seed': seed, 'checkpoint_exact': True,
                                               'all_epoch_history_exact': True, 'metrics_exact': True})
                        legacy.write_json(destination / 'baseline_parity.json', parity_results)
                base = checkpoint_state(paths['R1'])
                treatment = checkpoint_state(paths['R2'])
                for name, value in base.items():
                    if not (name.endswith(('.weight', '.bias')) or name == 'q_logits'):
                        continue
                    delta = (treatment[name] - value).double()
                    comparisons.append({'mode': mode, 'seed': seed, 'parameter': name,
                                        'shape': str(list(value.shape)), 'difference_l2': float(delta.norm()),
                                        'difference_max': float(delta.abs().max()),
                                        'relative_difference_l2': float(delta.norm() / value.double().norm().clamp_min(1e-30)),
                                        'effective_abs_weight_difference_l2': float((treatment[name].abs() - value.abs()).double().norm())
                                        if name.endswith('.weight') and name.startswith(('f_nn.', 'theta_agg_mat.')) else None})
                core.save_csv(destination / 'parameter_comparison.csv', comparisons)
            print(f'COMPLETED seed={seed}: both Q settings, R0/R1/R2; parity exact', flush=True)
        for filename, digest in hashes.items():
            assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest
        for filename, digest in data['source_sha256'].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
        legacy.write_json(destination / 'completed.json', {'status': 'passed', 'runs': len(records),
                          'source_and_data_unchanged': True, 'baseline_parity_pairs': len(parity_results)})
    except BaseException:
        (destination / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def controlled_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-id', default='student_cl_control_math1_20261123')
    p.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    p.add_argument('--lambda-cl', type=float, choices=[0.1], default=0.1)
    p.add_argument('--positive-noise-ratio', type=float, default=.1)
    p.add_argument('--positive-noise-scale', type=float, default=.01)
    p.add_argument('--negative-noise-ratio', type=float, default=.5)
    p.add_argument('--negative-noise-scale', type=float, default=.1)
    p.add_argument('--noise-distribution', choices=['uniform'], default='uniform')
    p.add_argument('--contrastive-seed', type=int, default=1123)
    p.add_argument('--temperature', type=float, default=.2)
    args = p.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in ('.', '..'):
        p.error('experiment-id must be one folder name')
    if args.contrastive_seed < 0 or any(s < 0 for s in args.seeds) or args.temperature <= 0:
        p.error('Invalid seed or temperature')
    if not 0 <= args.positive_noise_ratio < args.negative_noise_ratio <= 1:
        p.error('Require positive ratio < negative ratio')
    if not 0 <= args.positive_noise_scale < args.negative_noise_scale < 1:
        p.error('Require positive scale < negative scale < 1')
    args.dataset = 'math1'
    args.data_dir = str(ROOT / 'Q_GNCDM/data/math1')
    args.alpha, args.embedding_dim = .8, 32
    args.epochs, args.patience, args.batch_students = 30, 8, 128
    args.query_fraction = .25
    args.lr, args.q_lr = .001, .005
    args.lambda_q, args.lambda_coverage = .01, .01
    args.q_init_high, args.q_init_low = .55, .45
    args.device = 'cpu'
    return args


if __name__ == '__main__':
    run_experiment(controlled_args())


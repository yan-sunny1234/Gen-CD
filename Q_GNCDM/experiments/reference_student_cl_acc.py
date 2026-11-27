"""Reference original training loop, ACC selection and compact artifacts only.
Generated from run_simple_contrastive.py at git 89290e14293fe00b1e4eb55ba87ed068c5eccfdc.
Model, optimizer, batching, losses and evaluation are reused unchanged.
"""
from Q_GNCDM.experiments.run_simple_contrastive import *

def train_one(data, args, mode, seed, output):
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
            "seen-student; clean support for prediction; response-flipped support "
            "only for S1 InfoNCE; query hidden from student and item inputs"
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

            if settings["contrastive"]:
                noisy_np = flip_observed_responses(
                    clean_np,
                    args.positive_noise_rate,
                    np.random.default_rng([seed, epoch, step, 2]),
                )
                theta_noisy = model.diagnose_theta(torch.as_tensor(
                    noisy_np, dtype=torch.float32, device=model.device
                ))
                loss_cl = simple_infonce(
                    theta_clean, theta_noisy, args.temperature
                )
                flip_fraction = float(
                    np.sum(noisy_np != clean_np) / np.maximum(np.sum(clean_np != 0), 1)
                )
                coefficient = args.lambda_cl
            else:
                loss_cl = theta_clean.sum() * 0.0
                flip_fraction = 0.0
                coefficient = 0.0
            total = losses["loss_total"] + coefficient * loss_cl
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
                "loss_cl": float(loss_cl.detach().cpu()),
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
    if args.dataset == "synthetic":
        mastery_true = np.load(data["folder"] / "mastery_true.npy")
        result.update(legacy.mastery_metrics(theta, mastery_true))
    legacy.write_json(output / "metrics.json", result)
    return result


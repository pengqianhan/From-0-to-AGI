"""把阶梯实验的全部结果汇成一个自包含的 HTML 报告：runs/ladder-3090/report.html。

    uv run python runs/ladder-3090/build_report.py

读 results/*.csv、results/*.json（与 analyze.py 同一套口径），把数据嵌进 report_template.html。
README 里只以文字出现的数字（时间线、GPU·小时、数据量、试点吞吐）写在本文件的 MANUAL 里，出处注在旁边。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as A  # noqa: E402

SCALES = ["e5m", "e11m", "e24m", "e44m", "e83m"]
OT = ["e11m_ot", "e24m_ot"]
LOOP = ["e24m_lf3", "e24m_lm3", "e44m_lf3", "e44m_lm3"]

# README 里的数字（各条注明出处节）
MANUAL = {
    # 第 3 节"阶梯尺寸"
    "shapes": {
        "e5m": {"dim": 64, "layers": 2, "heads": "1 / 1", "ffn": 192, "nonemb": 0.11e6, "role": "拟合"},
        "e11m": {"dim": 128, "layers": 4, "heads": "2 / 1", "ffn": 320, "nonemb": 0.69e6, "role": "拟合"},
        "e24m": {"dim": 256, "layers": 4, "heads": "4 / 2", "ffn": 704, "nonemb": 2.95e6, "role": "拟合"},
        "e44m": {"dim": 384, "layers": 6, "heads": "6 / 3", "ffn": 1024, "nonemb": 9.74e6, "role": "拟合"},
        "e83m": {"dim": 512, "layers": 10, "heads": "8 / 4", "ffn": 1344, "nonemb": 28.5e6, "role": "留出检验"},
    },
    # 第 1 阶段试点（tok/s，单进程 micro 4）
    "pilot_toks": {"e5m": 218e3, "e11m": 180e3, "e24m": 143e3, "e44m": 105e3, "e83m": 70e3},
    # 试点后修正的计划 vs 各档"耗时"条目（GPU·小时）
    "gpu_hours": [
        {"stage": "e5m", "plan": 33, "actual": 60, "note": "主搜索 49 + 边界扩展 11"},
        {"stage": "e11m", "plan": 37, "actual": 72.9, "note": "含 7 组误重训、补搜多训满约 22 组"},
        {"stage": "e24m", "plan": 54, "actual": 61, "note": "暂停前约 40 + 最后一轮 10.2 + 衰减分叉 10.4"},
        {"stage": "e44m", "plan": 79, "actual": 72, "note": ""},
        {"stage": "e83m", "plan": 130, "actual": 109, "note": "主干约 83 + 衰减分叉 25.7"},
        {"stage": "过训练", "plan": 16, "actual": 16.6, "note": "只用 GPU0"},
        {"stage": "循环", "plan": 29, "actual": 29.8, "note": "e24m 12.7 + e44m 17.1；只用 GPU0"},
    ],
    # 第 0 阶段：训练数据（B token）与验证集 字节/token
    "data_tokens": [
        {"src": "FineWeb-Edu", "lang": "英文", "tokens": 1.261, "bpt": 4.51},
        {"src": "FineWeb-2 中文", "lang": "中文", "tokens": 0.817, "bpt": 4.38},
        {"src": "UltraData-Code", "lang": "代码", "tokens": 0.516, "bpt": 3.46},
        {"src": "FineMath", "lang": "英文", "tokens": 0.472, "bpt": 3.57},
        {"src": "DCLM", "lang": "英文", "tokens": 0.382, "bpt": 4.37},
        {"src": "Ultra-FineWeb 中文", "lang": "中文", "tokens": 0.373, "bpt": 4.17},
    ],
    # 时间线（NZDT）。gpus：那段时间用的卡
    "timeline": [
        {"lane": "准备", "label": "第 0 阶段：下载、流水线、分词器", "start": "2026-10-01T00:00", "end": "2026-10-01T18:00", "kind": "prep"},
        {"lane": "准备", "label": "第 1 阶段：试点（GPU0）", "start": "2026-10-01T18:00", "end": "2026-10-01T23:00", "kind": "prep"},
        {"lane": "e5m", "label": "128 组 + 36 组边界扩展", "start": "2026-10-01T23:00", "end": "2026-10-02T21:49", "kind": "run"},
        {"lane": "e11m", "label": "96 组 + 24 组边界扩展", "start": "2026-10-02T21:49", "end": "2026-10-03T14:19", "kind": "run"},
        {"lane": "e11m", "label": "暂停（用户要用 GPU）", "start": "2026-10-03T14:19", "end": "2026-10-04T14:46", "kind": "pause"},
        {"lane": "e11m", "label": "继续", "start": "2026-10-04T14:46", "end": "2026-10-04T23:35", "kind": "run"},
        {"lane": "e24m", "label": "48 组", "start": "2026-10-04T23:35", "end": "2026-10-05T15:12", "kind": "run"},
        {"lane": "e24m", "label": "暂停约 3 周", "start": "2026-10-05T15:12", "end": "2026-10-27T10:26", "kind": "pause"},
        {"lane": "e24m", "label": "最后一轮 + 衰减分叉（GPU0）", "start": "2026-10-27T10:26", "end": "2026-10-28T07:10", "kind": "run"},
        {"lane": "e24m", "label": "调度器崩溃（analyze.py 空集）", "start": "2026-10-28T07:10", "end": "2026-10-28T16:15", "kind": "crash"},
        {"lane": "e44m", "label": "24 组（GPU0）", "start": "2026-10-28T16:19", "end": "2026-10-31T16:17", "kind": "run"},
        {"lane": "e83m", "label": "6 组留出（GPU0）", "start": "2026-10-31T16:17", "end": "2026-11-03T09:17", "kind": "run"},
        {"lane": "e83m", "label": "GPU 维护", "start": "2026-11-03T09:17", "end": "2026-11-16T14:10", "kind": "pause"},
        {"lane": "e83m", "label": "继续 + 拟合", "start": "2026-11-16T14:10", "end": "2026-11-18T09:53", "kind": "run"},
        {"lane": "过训练", "label": "e11m / e24m 过训练（GPU0）", "start": "2026-11-23T10:21", "end": "2026-11-24T02:55", "kind": "run"},
        {"lane": "循环", "label": "e24m 两种结构（学习率检查 + 训满）", "start": "2026-11-24T14:42", "end": "2026-11-25T03:22", "kind": "run"},
        {"lane": "循环", "label": "e44m 两种结构", "start": "2026-11-25T03:22", "end": "2026-11-25T20:31", "kind": "run"},
    ],
    "gpus": [
        {"start": "2026-10-01T23:00", "end": "2026-10-03T13:35", "cards": "0,1,2"},
        {"start": "2026-10-03T13:35", "end": "2026-10-03T14:19", "cards": "0,2"},
        {"start": "2026-10-04T14:46", "end": "2026-10-05T10:00", "cards": "0,1,2"},
        {"start": "2026-10-05T10:00", "end": "2026-10-05T15:12", "cards": "0,2"},
        {"start": "2026-10-27T10:26", "end": "2026-11-25T20:31", "cards": "0"},
    ],
    # 主线：N_eff（序列 2048 口径）与 token
    "main": {"n_eff": 924319744.0, "tokens": 400e9},
    # 第 7 阶段开训前的试点：同一档稠密模型的吞吐（tok/s），与循环模型同一次测的
    "loop_dense_toks": {"e24m": 139945, "e44m": 101540},
}


def r4(x: float) -> float:
    return round(float(x), 5)


def law_dict(law) -> dict:  # noqa: ANN001
    return {"E": float(law.E), "A": float(law.A), "B": float(law.B), "alpha": float(law.alpha), "beta": float(law.beta)}


def main() -> None:
    from zero.tools.fit_scaling import fit_chinchilla

    data: dict = {"manual": MANUAL, "scales": {}, "ot": {}, "loop": {}}
    for s in SCALES + OT + LOOP:
        rows = A.load(s)
        cfg_ids = sorted({r["id"] for r in rows}, key=lambda i: (len(i), i))
        idx = {c: i for i, c in enumerate(cfg_ids)}
        first = {}
        for r in rows:
            first.setdefault(r["id"], r)
        configs = [{"id": c, "space": first[c].get("space", "full"),
                    **{h: float(first[c][h]) for h in A.HP},
                    **({"lr_mult": float(first[c]["lr_mult"])} if first[c].get("lr_mult") else {})}
                   for c in cfg_ids]
        # 同一个 (id, phase, k) 可能因为续训出现两次：取最后一条（与 analyze 一致）
        last: dict = {}
        for r in rows:
            last[(r["id"], r["phase"], r["k"])] = r
        pts = [[idx[r["id"]], r["k"], 1 if r["phase"] == "decay" else 0, int(r["tokens"]), r4(r["val_bpb"])]
               for r in last.values()]
        entry = {"n_eff": float(rows[0]["n_eff"]), "configs": configs, "rows": pts}
        if s in SCALES:
            fr = json.loads((A.RES / f"{s}_frontier.json").read_text())
            entry["frontier"] = {k: fr[k] for k in ("configs", "finished", "best_all_id", "best_all_val_bpb",
                                                    "best_all_config", "quantiles", "successive_halving")
                                 if k in fr}
            entry["points"] = [{"name": p.name, "D": p.D, "loss": p.loss} for p in A.frontier_points(s)]
            data["scales"][s] = entry
        elif s in LOOP:
            data["loop"][s] = entry
        else:
            entry["points"] = [{"name": p.name, "D": p.D, "loss": p.loss} for p in A.ot_points(s.removesuffix("_ot"))]
            data["ot"][s] = entry

    # 三次拟合：预注册（e5m–e44m）、去掉 e5m、去掉 e5m + 过训练点
    sl = json.loads((A.RES / "scaling_law.json").read_text())
    drop = fit_chinchilla([p for s in ["e11m", "e24m", "e44m"] for p in A.frontier_points(s)], param_count="flops")
    ot = json.loads((A.RES / "scaling_law_ot.json").read_text())
    diag = json.loads((A.RES / "fit_diagnosis.json").read_text())
    data["fits"] = {
        "prereg": {"label": "e5m–e44m（第 4 阶段预注册）", "law": {k: sl["law"][k] for k in "E A B alpha beta".split()},
                   "holdout": sl["holdout_rel_err"], "fit_max": sl["fit_max_rel_err"],
                   "main": sl["main_T2048"]["val_bpb"], "ci": sl["main_T2048"]["ci95"], "ci_kind": "bootstrap 95%"},
        "drop": {"label": "去掉 e5m（事后诊断，= 第 6 阶段预注册）", "law": law_dict(drop),
                 "holdout": dict(zip(sl["holdout_rel_err"], diag["fixed_E"]["0.0"]["holdout_rel_err"])),
                 "fit_max": diag["variants"]["drop_e5m"]["fit_max_rel_err"],
                 "main": diag["variants"]["drop_e5m"]["main_T2048_400B"], "ci": diag["drop_e5m_main_ci95"],
                 "ci_kind": "bootstrap（3 个 N 组）"},
        "ot": {"label": "去掉 e5m + 4 个过训练点（只报告）", "law": {k: ot["law"][k] for k in "E A B alpha beta".split()},
               "holdout": ot["holdout_rel_err"], "fit_max": ot["law"]["max_rel_err"],
               "main": ot["main_T2048_400B"], "ci": ot["main_ci95"], "ci_kind": "去掉某一档的敏感度"},
    }
    data["diag"] = {"variants": diag["variants"], "fixed_E": diag["fixed_E"]}
    data["ot_fit"] = {k: ot[k] for k in ("E_profile_rms", "E_allowed", "preregistered_pred_rel_err", "gates", "new_points")}
    data["ot_pred"] = json.loads((A.RES / "ot_predictions.json").read_text())
    data["loop_result"] = json.loads((A.RES / "loop_result.json").read_text())
    data["decisions"] = json.loads((A.RES / "decisions.json").read_text())
    for s in ["e5m", "e11m"]:
        data["scales"][s]["nql"] = json.loads((A.RES / f"{s}_nql.json").read_text())
    data["narrow"] = {s: json.loads((A.RES / f"{s}_narrow_ranges.json").read_text())["ranges"]
                      for s in ["e24m", "e44m", "e83m"]}

    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=float).replace("</", "<\\/")
    tpl = (HERE / "report_template.html").read_text()
    out = HERE / "report.html"
    out.write_text(tpl.replace("/*__DATA__*/null", blob))
    print(f"{out}：{out.stat().st_size / 1024:.0f} KB（数据 {len(blob) / 1024:.0f} KB）")


if __name__ == "__main__":
    main()

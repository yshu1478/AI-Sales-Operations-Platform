from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from styles.theme import (
    PLOTLY_COLORWAY,
    PLOTLY_SEQUENTIAL,
    apply_plotly_theme,
    load_theme,
    render_page_header,
    render_sidebar_brand,
    render_timeline_item,
)


TODAY = pd.Timestamp(date.today())
STAGE_ORDER = ["初步接洽", "需求确认", "方案报价", "商务谈判", "已成交", "已流失"]
ACTIVE_STAGES = STAGE_ORDER[:5]
DATA_COLUMNS = ["客户名称", "客户行业", "获客渠道", "商机阶段", "预计成交金额", "负责人", "最近跟进日期", "是否成交"]
REQUIRED_COLUMNS = DATA_COLUMNS[:-1]
DETAIL_STAGES = {"初步接洽": 0.15, "需求确认": 0.30, "方案报价": 0.55, "商务谈判": 0.75, "已成交": 1.00, "已流失": 0.00}


@st.cache_data
def generate_sales_data(rows: int = 240, seed: int = 20260717) -> pd.DataFrame:
    """生成仅用于演示的、可复现的模拟销售商机数据。"""
    rng = np.random.default_rng(seed)
    industries = ["互联网", "制造业", "零售", "金融", "医疗健康", "教育"]
    channels = ["官网咨询", "老客推荐", "线下活动", "内容营销", "合作伙伴", "电话拓客"]
    owners = ["张伟", "李娜", "王芳", "刘洋", "陈晨", "赵磊"]
    stages = rng.choice(STAGE_ORDER, rows, p=[0.18, 0.20, 0.18, 0.14, 0.20, 0.10])
    channel_values = rng.choice(channels, rows, p=[0.18, 0.23, 0.15, 0.16, 0.17, 0.11])
    base_amount = rng.lognormal(mean=11.0, sigma=0.75, size=rows)
    stage_factor = pd.Series(stages).map(
        {"初步接洽": 0.75, "需求确认": 0.9, "方案报价": 1.05, "商务谈判": 1.2, "已成交": 1.15, "已流失": 0.85}
    ).to_numpy()

    return pd.DataFrame(
        {
            "客户名称": [f"模拟客户{i:03d}" for i in range(1, rows + 1)],
            "客户行业": rng.choice(industries, rows),
            "获客渠道": channel_values,
            "商机阶段": stages,
            "预计成交金额": np.round(base_amount * stage_factor / 1000) * 1000,
            "负责人": rng.choice(owners, rows),
            "最近跟进日期": TODAY - pd.to_timedelta(rng.integers(0, 31, rows), unit="D"),
            "是否成交": np.where(stages == "已成交", "是", "否"),
        }
    )


def initialize_database(db_path: str | Path, initial_data: pd.DataFrame) -> None:
    """首次运行时建立商机库，并用演示数据初始化；之后不会覆盖已保存数据。"""
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            '''CREATE TABLE IF NOT EXISTS opportunities (
                "商机ID" INTEGER PRIMARY KEY AUTOINCREMENT,
                "客户名称" TEXT NOT NULL,
                "客户行业" TEXT NOT NULL,
                "获客渠道" TEXT NOT NULL,
                "商机阶段" TEXT NOT NULL,
                "预计成交金额" REAL NOT NULL,
                "负责人" TEXT NOT NULL,
                "最近跟进日期" TEXT NOT NULL,
                "是否成交" TEXT NOT NULL
            )'''
        )
        count = connection.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]
        if count == 0:
            prepared = initial_data[DATA_COLUMNS].copy()
            prepared["最近跟进日期"] = pd.to_datetime(prepared["最近跟进日期"]).dt.strftime("%Y-%m-%d")
            prepared.to_sql("opportunities", connection, if_exists="append", index=False)


def load_opportunities(db_path: str | Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as connection:
        data = pd.read_sql_query('SELECT * FROM opportunities ORDER BY "商机ID"', connection)
    data["最近跟进日期"] = pd.to_datetime(data["最近跟进日期"])
    return data


def validate_opportunities(data: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        return [f"缺少字段：{', '.join(missing_columns)}"]
    blank_text = data[REQUIRED_COLUMNS].astype(str).apply(lambda column: column.str.strip().eq(""))
    if data[REQUIRED_COLUMNS].isna().any().any() or blank_text.any().any():
        errors.append("客户名称、行业、渠道、阶段、金额、负责人和跟进日期均不能为空。")
    amounts = pd.to_numeric(data["预计成交金额"], errors="coerce")
    if amounts.isna().any() or (amounts < 0).any():
        errors.append("预计成交金额必须是大于或等于 0 的数字。")
    invalid_stages = sorted(set(data["商机阶段"].dropna()) - set(STAGE_ORDER))
    if invalid_stages:
        errors.append(f"存在无效商机阶段：{', '.join(invalid_stages)}")
    return errors


def save_opportunities(db_path: str | Path, data: pd.DataFrame) -> None:
    """事务式保存完整台账；验证失败时数据库保持不变。"""
    errors = validate_opportunities(data)
    if errors:
        raise ValueError("；".join(errors))
    prepared = data.copy()
    prepared["预计成交金额"] = pd.to_numeric(prepared["预计成交金额"])
    prepared["最近跟进日期"] = pd.to_datetime(prepared["最近跟进日期"]).dt.strftime("%Y-%m-%d")
    prepared["是否成交"] = np.where(prepared["商机阶段"] == "已成交", "是", "否")
    prepared = prepared[DATA_COLUMNS]
    with sqlite3.connect(db_path) as connection:
        connection.execute("DELETE FROM opportunities")
        prepared.to_sql("opportunities", connection, if_exists="append", index=False)


def build_customer_detail(opportunity: pd.Series | dict) -> dict:
    """根据商机记录确定性生成客户详情演示数据，不使用真实联系人或隐私数据。"""
    record = opportunity.to_dict() if isinstance(opportunity, pd.Series) else dict(opportunity)
    seed = int(hashlib.sha256(str(record["客户名称"]).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    company_sizes = ["1-50人", "51-200人", "201-500人", "501-1000人", "1000人以上"]
    regions = ["华北·北京", "华东·上海", "华南·深圳", "华中·武汉", "西南·成都"]
    contact_names = ["刘经理", "王老师", "陈总", "赵女士", "李经理"]
    contact_titles = ["采购负责人", "信息化负责人", "销售总监", "运营负责人", "企业服务经理"]
    last_followup = pd.Timestamp(record["最近跟进日期"])
    stage = record["商机阶段"]
    expected_date = last_followup + pd.to_timedelta(int(rng.integers(14, 60)), unit="D")
    next_followup = pd.Timestamp(TODAY + pd.to_timedelta(int(rng.integers(1, 10)), unit="D"))
    contact_index = int(rng.integers(0, len(contact_names)))
    records = []
    activities = [(last_followup - pd.to_timedelta(12, unit="D"), "完成首次需求沟通，确认业务团队的核心使用场景。"), (last_followup - pd.to_timedelta(7, unit="D"), "向客户演示产品方案，并收集采购流程和预算信息。"), (last_followup - pd.to_timedelta(3, unit="D"), "同步方案调整项，约定下一次会议和内部评审时间。"), (last_followup, "完成本阶段跟进，已更新商机阶段和预计成交计划。")]
    for activity_date, content in activities:
        records.append({"时间": activity_date, "类型": "销售跟进", "内容": content})
    return {"基本信息": {"客户名称": record["客户名称"], "行业": record["客户行业"], "公司规模": company_sizes[int(rng.integers(0, len(company_sizes)))], "获客渠道": record["获客渠道"], "所在地区": regions[int(rng.integers(0, len(regions)))]}, "联系人": {"姓名": contact_names[contact_index], "职位": contact_titles[contact_index], "电话": f"138****{int(rng.integers(1000, 9999)):04d}", "邮箱": f"contact{int(record['商机ID']):03d}@example.demo"}, "商机信息": {"商机阶段": stage, "成交概率": DETAIL_STAGES.get(stage, 0.0), "预计成交金额": float(record["预计成交金额"]), "预计成交日期": expected_date}, "销售跟进": {"最近跟进时间": last_followup, "下一次跟进时间": next_followup, "跟进记录": records}}


def ai_risk_prediction(opportunity: pd.Series | dict, all_data: pd.DataFrame) -> dict:
    """基于当前商机台账计算可解释的风险评分，不调用固定文案或外部数据。"""
    record = opportunity.to_dict() if isinstance(opportunity, pd.Series) else dict(opportunity)
    days_since_followup = max(0, (TODAY - pd.Timestamp(record["最近跟进日期"])).days)
    amount = float(record["预计成交金额"])
    median_amount = float(pd.to_numeric(all_data["预计成交金额"], errors="coerce").median()) if not all_data.empty else amount
    score = 0
    reasons: list[str] = []
    if record["商机阶段"] in {"已成交", "已流失"}:
        return {"score": 0 if record["商机阶段"] == "已成交" else 100, "level": "已关闭", "reasons": ["商机已结束，不再进入在途风险计算"]}
    if days_since_followup > 7:
        score += min(45, 20 + (days_since_followup - 7) * 2)
        reasons.append(f"已连续 {days_since_followup} 天未跟进")
    if days_since_followup > 14:
        score += 15
        reasons.append("跟进间隔超过两周，客户热度可能下降")
    if amount >= median_amount * 1.5:
        score += 15
        reasons.append(f"金额 ¥{amount:,.0f} 高于台账中位数的 1.5 倍")
    if record["商机阶段"] == "初步接洽":
        score += 10
        reasons.append("仍处于早期阶段，需求与决策链尚未充分验证")
    score = min(100, score)
    level = "高风险" if score >= 60 else "中风险" if score >= 35 else "低风险"
    if not reasons:
        reasons.append("跟进节奏、金额和阶段暂未触发明显风险信号")
    return {"score": score, "level": level, "reasons": reasons}


def ai_followup_suggestion(opportunity: pd.Series | dict, all_data: pd.DataFrame) -> list[str]:
    record = opportunity.to_dict() if isinstance(opportunity, pd.Series) else dict(opportunity)
    risk = ai_risk_prediction(record, all_data)
    days = max(0, (TODAY - pd.Timestamp(record["最近跟进日期"])).days)
    suggestions = []
    if record["商机阶段"] == "初步接洽":
        suggestions.append(f"围绕 {record['客户行业']} 行业的关键业务场景发起一次 20 分钟需求访谈，目标是确认决策人和预算窗口。")
    elif record["商机阶段"] == "需求确认":
        suggestions.append("整理客户已确认的需求清单，邀请业务负责人参与方案共创，推动进入报价评审。")
    elif record["商机阶段"] == "方案报价":
        suggestions.append(f"针对 ¥{float(record['预计成交金额']):,.0f} 商机安排方案复盘，明确采购流程、竞争对手和最终决策日期。")
    elif record["商机阶段"] == "商务谈判":
        suggestions.append("确认合同、付款和上线条款的最后阻塞点，并约定双方签署时间，不再只做泛泛跟进。")
    if days > 7:
        suggestions.append(f"该客户已 {days} 天未跟进，建议今天完成一次带明确问题和下一步日期的触达。")
    if risk["score"] >= 60:
        suggestions.append("风险较高：建议销售主管参与下一次沟通，并在 CRM 中补齐决策链和预计成交日期。")
    if not suggestions:
        suggestions.append("保持当前跟进节奏，下一次沟通后及时记录客户反馈和明确的下一步动作。")
    return suggestions


def ai_customer_profile(opportunity: pd.Series | dict, all_data: pd.DataFrame) -> dict:
    record = opportunity.to_dict() if isinstance(opportunity, pd.Series) else dict(opportunity)
    segment = all_data[all_data["客户行业"] == record["客户行业"]]
    channel = all_data[all_data["获客渠道"] == record["获客渠道"]]
    channel_rate = float(channel["是否成交"].eq("是").mean()) if not channel.empty else 0.0
    owner_pipeline = float(all_data.loc[(all_data["负责人"] == record["负责人"]) & (~all_data["商机阶段"].isin(["已成交", "已流失"])), "预计成交金额"].sum())
    return {"定位": f"{record['客户行业']}行业客户，来自{record['获客渠道']}，当前处于{record['商机阶段']}。", "同业商机数": len(segment), "来源转化率": channel_rate, "负责人在途金额": owner_pipeline, "风险": ai_risk_prediction(record, all_data)}


def build_ai_analysis_prompt(customer: pd.Series | dict, detail: dict) -> str:
    """构造发送给兼容 OpenAI Chat Completions 接口的结构化提示词。"""
    record = customer.to_dict() if isinstance(customer, pd.Series) else dict(customer)
    safe_record = {key: (value.isoformat() if isinstance(value, (pd.Timestamp, date)) else value) for key, value in record.items() if key != "商机ID"}
    context = {"客户记录": safe_record, "客户详情": detail}
    output_schema = {
        "summary": "一句话总结",
        "probability": 55,
        "risk_level": "高/中/低",
        "risk_reasons": ["原因1", "原因2"],
        "next_actions": ["建议1", "建议2"],
        "contact_time": "建议联系时间",
        "contact_person": "建议联系对象",
    }
    return f"""你是一名企业 ToB 销售运营经理。请只根据下面的当前客户与商机信息进行分析，不要编造未提供的事实。

客户与商机数据：
{json.dumps(context, ensure_ascii=False, default=str)}

请严格遵守以下要求：
1. 只返回一个合法 JSON 对象，不要输出 Markdown 代码围栏、解释文字或额外字段。
2. probability 必须是 0 到 100 之间的数字。
3. risk_level 只能是“高”“中”“低”之一。
4. risk_reasons 和 next_actions 必须是字符串数组；建议应具体、可执行，并基于已提供的数据。
5. 如果信息不足，请明确写入 summary 或 risk_reasons，不要猜测。

返回格式：
{json.dumps(output_schema, ensure_ascii=False, indent=2)}"""


def parse_ai_analysis_response(content: str) -> dict:
    """解析并校验 LLM JSON；无法结构化时保留原始文本供页面展示。"""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        required = {"summary", "probability", "risk_level", "risk_reasons", "next_actions", "contact_time", "contact_person"}
        if not isinstance(parsed, dict) or not required.issubset(parsed):
            raise ValueError("返回 JSON 缺少必需字段")
        probability = float(parsed["probability"])
        if not 0 <= probability <= 100:
            raise ValueError("成交概率必须在 0 到 100 之间")
        risk_level = str(parsed["risk_level"]).replace("风险", "").strip()
        if risk_level not in {"高", "中", "低"}:
            raise ValueError("风险等级必须为高、中或低")
        if not isinstance(parsed["risk_reasons"], list) or not isinstance(parsed["next_actions"], list):
            raise ValueError("风险原因和下一步建议必须为数组")
        return {
            "summary": str(parsed["summary"]),
            "probability": probability,
            "risk_level": risk_level,
            "risk_reasons": [str(item) for item in parsed["risk_reasons"]],
            "next_actions": [str(item) for item in parsed["next_actions"]],
            "contact_time": str(parsed["contact_time"]),
            "contact_person": str(parsed["contact_person"]),
        }
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"raw_text": content.strip() or "DeepSeek 未返回分析内容。"}


def request_deepseek_analysis(customer: pd.Series | dict, detail: dict) -> dict:
    """通过 DeepSeek OpenAI 兼容接口分析当前客户；API Key 只从环境变量读取。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("尚未配置 DeepSeek API Key。请先设置 DEEPSEEK_API_KEY 环境变量，再重新点击“AI分析”。")
    try:
        from openai import APITimeoutError, OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 依赖，请先执行 pip install -r requirements.txt") from exc
    client = OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), timeout=30.0)
    request_args = {"model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), "messages": [{"role": "system", "content": "你输出简洁、可执行、基于事实的销售运营分析。"}, {"role": "user", "content": build_ai_analysis_prompt(customer, detail)}], "temperature": 0.2, "response_format": {"type": "json_object"}}
    try:
        response = client.chat.completions.create(**request_args)
    except (APITimeoutError, TimeoutError) as exc:
        raise RuntimeError("DeepSeek API 请求超时，请检查网络后稍后重试。") from exc
    except Exception as first_error:
        # 部分兼容服务不支持 response_format，降级重试一次，仍使用同一客户上下文。
        request_args.pop("response_format", None)
        try:
            response = client.chat.completions.create(**request_args)
        except (APITimeoutError, TimeoutError) as exc:
            raise RuntimeError("DeepSeek API 请求超时，请检查网络后稍后重试。") from exc
        except Exception as second_error:
            raise RuntimeError(f"DeepSeek 请求失败：{second_error}") from first_error
    content = response.choices[0].message.content or ""
    return parse_ai_analysis_response(content)


def ai_sales_daily_report(data: pd.DataFrame) -> dict:
    recent = data[(TODAY - pd.to_datetime(data["最近跟进日期"])).dt.days <= 7]
    overdue = data[(data["是否成交"] == "否") & ((TODAY - pd.to_datetime(data["最近跟进日期"])).dt.days > 7)]
    won_recent = data[(data["是否成交"] == "是") & ((TODAY - pd.to_datetime(data["最近跟进日期"])).dt.days <= 7)]
    owner = data.groupby("负责人")["预计成交金额"].sum().sort_values(ascending=False)
    return {"跟进商机数": len(recent), "近7天成交金额": float(won_recent["预计成交金额"].sum()), "超期商机数": len(overdue), "重点负责人": owner.index[0] if not owner.empty else "暂无", "重点负责人金额": float(owner.iloc[0]) if not owner.empty else 0.0}


def ai_sales_analysis(data: pd.DataFrame) -> dict:
    summary = channel_summary(data)
    stage_counts = data["商机阶段"].value_counts()
    top_channel = summary.iloc[0]["获客渠道"] if not summary.empty else "暂无"
    best_channel = summary.sort_values("转化率", ascending=False).iloc[0]["获客渠道"] if not summary.empty else "暂无"
    bottleneck = stage_counts.index[0] if not stage_counts.empty else "暂无"
    risk_count = sum(ai_risk_prediction(row, data)["level"] == "高风险" for _, row in data.iterrows())
    return {"商机最多渠道": top_channel, "转化率最高渠道": best_channel, "最大阶段": bottleneck, "高风险商机数": risk_count, "渠道摘要": summary}


def generate_ai_action_recommendations(data: pd.DataFrame) -> list[dict[str, str]]:
    """从当前台账指标推导行动建议；每次数据变化都会重新计算建议内容和优先级。"""
    if data.empty:
        return [{"建议": "先补充商机台账数据后再生成行动建议。", "为什么": "当前没有可分析的商机数量、转化率和金额数据。", "业务帮助": "避免在缺少事实依据时做出错误销售决策。", "优先级": "高"}]
    working = data.copy()
    working["跟进天数"] = (TODAY - pd.to_datetime(working["最近跟进日期"])).dt.days.clip(lower=0)
    open_data = working[~working["商机阶段"].isin(["已成交", "已流失"])].copy()
    overdue = open_data[open_data["跟进天数"] > 7]
    total = len(data)
    conversion = float(data["是否成交"].eq("是").mean())
    pipeline = float(open_data["预计成交金额"].sum())
    overdue_amount = float(overdue["预计成交金额"].sum())
    top10_amount = float(overdue.nlargest(10, "预计成交金额")["预计成交金额"].sum())
    owner_stats = open_data.groupby("负责人", as_index=False).agg(在途金额=("预计成交金额", "sum"), 商机数=("客户名称", "size"))
    owner_overdue = overdue.groupby("负责人", as_index=False).size().rename(columns={"size": "超期数"})
    owner_stats = owner_stats.merge(owner_overdue, on="负责人", how="left").fillna({"超期数": 0})
    owner_stats["超期率"] = owner_stats["超期数"] / owner_stats["商机数"].replace(0, 1)
    if owner_stats.empty:
        focus_owner, focus_owner_amount, focus_owner_overdue = "暂无负责人", 0.0, 0
    else:
        focus = owner_stats.sort_values(["超期数", "在途金额"], ascending=False).iloc[0]
        focus_owner, focus_owner_amount, focus_owner_overdue = focus["负责人"], float(focus["在途金额"]), int(focus["超期数"])
    channel = channel_summary(data)
    if channel.empty:
        best_channel = worst_channel = "暂无渠道"
        best_rate = worst_rate = 0.0
    else:
        best = channel.sort_values(["转化率", "商机数量"], ascending=False).iloc[0]
        worst = channel.sort_values(["转化率", "商机数量"], ascending=[True, False]).iloc[0]
        best_channel, best_rate = best["获客渠道"], float(best["转化率"])
        worst_channel, worst_rate = worst["获客渠道"], float(worst["转化率"])
    stage_counts = open_data["商机阶段"].value_counts()
    bottleneck = stage_counts.index[0] if not stage_counts.empty else "暂无阶段"
    bottleneck_count = int(stage_counts.iloc[0]) if not stage_counts.empty else 0
    weighted_pipeline = sum(float(row["预计成交金额"]) * DETAIL_STAGES.get(row["商机阶段"], 0) for _, row in open_data.iterrows())
    overdue_priority = "高" if len(overdue) >= max(5, int(total * 0.2)) else "中" if len(overdue) else "低"
    channel_priority = "高" if best_rate - worst_rate >= 0.15 else "中"
    recommendations = [
        {"建议": f"优先联系预计成交金额 Top10 的超期商机，并由 {focus_owner} 牵头完成今日触达。", "为什么": f"当前有 {len(overdue)} 条商机超过 7 天未跟进，涉及 ¥{overdue_amount:,.0f}；其中 Top10 就占 ¥{top10_amount:,.0f}，{focus_owner} 名下超期 {focus_owner_overdue} 条。", "业务帮助": "先保护高金额管道，减少客户热度下降造成的预计流失，明确当天责任人。", "优先级": overdue_priority},
        {"建议": f"为 {focus_owner} 安排一次管道复盘，逐条确认下一步动作和预计成交日期。", "为什么": f"该负责人当前在途金额 ¥{focus_owner_amount:,.0f}，同时有 {focus_owner_overdue} 条超期商机，负责人风险集中度最高。", "业务帮助": "把金额和责任人绑定，帮助主管优先清理最可能影响团队预测的个人管道。", "优先级": "高" if focus_owner_overdue >= 3 else "中"},
        {"建议": f"把新增线索更多分配给 {best_channel}，并复盘 {worst_channel} 的线索质量与跟进流程。", "为什么": f"整体转化率为 {conversion:.1%}；{best_channel} 转化率 {best_rate:.1%}，{worst_channel} 仅 {worst_rate:.1%}，渠道差距为 {(best_rate - worst_rate):.1%}。", "业务帮助": "提高获客预算和销售产能的投入产出比，避免低转化渠道持续消耗跟进资源。", "优先级": channel_priority},
        {"建议": f"围绕 {bottleneck} 阶段建立阶段推进清单，要求每条商机补齐下一步和客户决策人。", "为什么": f"当前在途商机中有 {bottleneck_count} 条集中在 {bottleneck}，占在途数量的 {bottleneck_count / max(len(open_data), 1):.1%}；整体转化率仅 {conversion:.1%}。", "业务帮助": "减少漏斗中段停滞，让销售经理能判断商机是真实推进还是虚假繁荣。", "优先级": "中" if bottleneck_count < len(open_data) * 0.5 else "高"},
        {"建议": f"用加权管道 ¥{weighted_pipeline:,.0f} 制定本周预测，并对照未加权在途金额 ¥{pipeline:,.0f} 做差异解释。", "为什么": f"当前总商机金额转化率为 {conversion:.1%}，阶段概率折算后可期待金额约为 ¥{weighted_pipeline:,.0f}，与在途金额差异反映了阶段质量。", "业务帮助": "让管理层用概率调整后的数字排产和预测，降低只看商机总额导致的乐观偏差。", "优先级": "高" if pipeline > 0 and weighted_pipeline / pipeline < 0.35 else "中"},
        {"建议": f"将超期未跟进率压到 7% 以下：本周由各负责人清理 {len(overdue)} 条超期商机，每条必须留下下一次跟进日期。", "为什么": f"当前超期商机 {len(overdue)} 条，占全部商机 {len(overdue) / max(total, 1):.1%}；负责人维度已出现跟进节奏不均。", "业务帮助": "建立可量化的跟进纪律，让 AI 风险预测有可靠的最新日期输入。", "优先级": "高" if len(overdue) else "低"},
    ]
    return recommendations


def render_ai_sales_insights(data: pd.DataFrame) -> None:
    report = ai_sales_daily_report(data)
    analysis = ai_sales_analysis(data)
    with st.expander("🤖 AI销售日报与分析", expanded=True):
        st.caption("AI结果基于当前 CRM 台账实时计算；修改或保存商机后刷新即可更新。")
        cols = st.columns(4)
        cols[0].metric("近7天跟进商机", f"{report['跟进商机数']} 条")
        cols[1].metric("近7天成交金额", f"¥{report['近7天成交金额']:,.0f}")
        cols[2].metric("超期未跟进", f"{report['超期商机数']} 条")
        cols[3].metric("重点负责人", report["重点负责人"], f"在途 ¥{report['重点负责人金额']:,.0f}")
        st.markdown(f"**AI销售分析**：当前商机数量最多的渠道是 **{analysis['商机最多渠道']}**，转化率最高的渠道是 **{analysis['转化率最高渠道']}**；商机主要集中在 **{analysis['最大阶段']}**，识别到 **{analysis['高风险商机数']}** 条高风险在途商机。")
        recommendations = generate_ai_action_recommendations(data)
        overdue_count = int(((data["是否成交"] == "否") & ((TODAY - pd.to_datetime(data["最近跟进日期"])).dt.days > 7)).sum())
        overdue_data = data[(data["是否成交"] == "否") & ((TODAY - pd.to_datetime(data["最近跟进日期"])).dt.days > 7)]
        if overdue_count:
            warning_level = "高" if overdue_count >= max(5, len(data) * 0.2) else "中"
            top_amount = float(overdue_data.nlargest(10, "预计成交金额")["预计成交金额"].sum())
            st.error(f"🚨 AI风险预警\n\n发现 **{overdue_count}** 条商机超过 7 天未跟进\n\n预计流失风险：**{warning_level}**\n\n建议：优先联系预计成交金额 Top10 客户（涉及 ¥{top_amount:,.0f}）。")
        st.markdown("### AI行动建议")
        for index, item in enumerate(recommendations, start=1):
            with st.container(border=True):
                st.markdown(f"**{index}. {item['建议']}**　`优先级：{item['优先级']}`")
                st.markdown(f"- **为什么提出**：{item['为什么']}\n- **业务帮助**：{item['业务帮助']}")


def render_customer_detail(data: pd.DataFrame) -> None:
    selected_id = st.session_state.get("detail_customer_id")
    valid_ids = set(data["商机ID"].astype(int))
    if selected_id is None or int(selected_id) not in valid_ids:
        selected_id = st.selectbox("选择客户", sorted(valid_ids), format_func=lambda value: data.loc[data["商机ID"] == value, "客户名称"].iloc[0])
    customer = data.loc[data["商机ID"].astype(int) == int(selected_id)].iloc[0]
    detail = build_customer_detail(customer)
    profile = ai_customer_profile(customer, data)
    render_page_header(
        f"{detail['基本信息']['客户名称']} · 客户详情",
        "CRM客户管理 / 客户详情",
        "集中查看客户资料、联系人、商机阶段、销售跟进与 AI 分析结果。",
    )
    if st.button("← 返回商机台账"):
        st.session_state["current_page"] = "商机台账管理"
        st.session_state.pop("detail_customer_id", None)
        st.rerun()

    risk = profile["风险"]
    with st.expander("🤖 AI客户画像、风险预测与跟进建议", expanded=True):
        st.markdown(f"**客户画像**：{profile['定位']}")
        profile_cols = st.columns(3)
        profile_cols[0].metric("同业商机数", f"{profile['同业商机数']} 条")
        profile_cols[1].metric("来源渠道转化率", f"{profile['来源转化率']:.1%}")
        profile_cols[2].metric("AI风险评分", f"{risk['score']} · {risk['level']}")
        st.markdown("**风险依据**：" + "；".join(risk["reasons"]))
        st.markdown("**AI跟进建议**")
        for suggestion in ai_followup_suggestion(customer, data):
            st.markdown(f"- {suggestion}")
        st.divider()
        st.markdown("**DeepSeek 实时 AI 分析**")
        st.caption("点击后会将当前客户与商机信息发送到你配置的 OpenAI 兼容接口。")
        loading_key = f"deepseek_analysis_loading_{int(selected_id)}"
        is_loading = bool(st.session_state.get(loading_key, False))
        analyze_clicked = st.button("🧠 AI分析", key=f"deepseek_analysis_{int(selected_id)}", type="primary", disabled=is_loading)
        if analyze_clicked and not is_loading:
            st.session_state[loading_key] = True
            with st.spinner("AI 正在分析客户与商机数据，请稍候……"):
                try:
                    st.session_state["llm_customer_analysis"] = request_deepseek_analysis(customer, detail)
                    st.session_state["llm_customer_analysis_id"] = int(selected_id)
                    st.session_state["llm_customer_analysis_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                except RuntimeError as exc:
                    st.error(str(exc))
                finally:
                    st.session_state[loading_key] = False
        llm_result = st.session_state.get("llm_customer_analysis")
        if llm_result and st.session_state.get("llm_customer_analysis_id") == int(selected_id):
            analysis_time = st.session_state.get("llm_customer_analysis_time")
            st.divider()
            st.markdown("#### AI 分析结果")
            if analysis_time:
                st.caption(f"分析时间：{analysis_time}")

            if "raw_text" in llm_result:
                st.warning("DeepSeek 返回的内容不是预期的 JSON 格式，已为你保留原始分析文本。")
                with st.container(border=True):
                    st.write(llm_result["raw_text"])
            elif {"summary", "probability", "risk_level", "risk_reasons", "next_actions", "contact_time", "contact_person"}.issubset(llm_result):
                with st.container(border=True):
                    st.markdown("**一句话总结**")
                    st.write(llm_result["summary"])

                metric_col, risk_col = st.columns(2)
                metric_col.metric("LLM 成交概率", f"{llm_result['probability']:.0f}%")
                risk_message = f"{llm_result['risk_level']}风险"
                if llm_result["risk_level"] == "高":
                    risk_col.error(f"🔴 **风险等级：{risk_message}**")
                elif llm_result["risk_level"] == "中":
                    risk_col.warning(f"🟠 **风险等级：{risk_message}**")
                else:
                    risk_col.success(f"🟢 **风险等级：{risk_message}**")

                st.markdown("**风险原因**")
                with st.container(border=True):
                    for reason in llm_result["risk_reasons"]:
                        st.markdown(f"- {reason}")

                st.markdown("**下一步销售建议**")
                actions_text = "\n".join(f"- {action}" for action in llm_result["next_actions"])
                st.info(actions_text or "DeepSeek 暂未给出下一步建议。")

                contact_time_col, contact_person_col = st.columns(2)
                contact_time_col.metric("推荐联系时间", llm_result["contact_time"])
                contact_person_col.metric("建议联系对象", llm_result["contact_person"])
            else:
                st.info("分析结果格式已升级，请重新点击“AI分析”生成最新结果。")
    st.subheader("基本信息")
    info_cols = st.columns(5)
    for col, (label, value) in zip(info_cols, detail["基本信息"].items()):
        col.metric(label, value)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("联系人")
            st.dataframe(pd.DataFrame([detail["联系人"]]), use_container_width=True, hide_index=True)
    with right:
        with st.container(border=True):
            st.subheader("商机信息")
            opportunity = detail["商机信息"].copy()
            opportunity["成交概率"] = f"{opportunity['成交概率']:.0%}"
            opportunity["预计成交金额"] = f"¥{opportunity['预计成交金额']:,.0f}"
            opportunity["预计成交日期"] = opportunity["预计成交日期"].strftime("%Y-%m-%d")
            st.dataframe(pd.DataFrame([opportunity]), use_container_width=True, hide_index=True)
    st.subheader("销售跟进")
    followup_cols = st.columns(2)
    followup_cols[0].metric("最近跟进时间", detail["销售跟进"]["最近跟进时间"].strftime("%Y-%m-%d"))
    followup_cols[1].metric("下一次跟进时间", detail["销售跟进"]["下一次跟进时间"].strftime("%Y-%m-%d"))
    for item in detail["销售跟进"]["跟进记录"]:
        render_timeline_item(item["时间"], item["类型"], item["内容"])


def render_opportunity_ledger(db_path: Path, data: pd.DataFrame) -> None:
    render_page_header(
        "商机台账管理",
        "CRM客户管理 / 商机台账",
        "可直接新增、编辑或删除商机；保存后写入本地数据库，重启应用数据仍会保留。",
    )
    st.subheader("客户列表")
    st.caption("点击客户所在行后，再点击“查看客户详情”进入客户详情页面。")
    picker_data = data[["商机ID", "客户名称", "客户行业", "获客渠道", "商机阶段", "预计成交金额"]].copy()
    risks = [ai_risk_prediction(row, data) for _, row in data.iterrows()]
    picker_data["AI风险"] = [f"{item['score']} · {item['level']}" for item in risks]
    picker = st.dataframe(picker_data, use_container_width=True, hide_index=True, selection_mode="single-row", on_select="rerun", key="customer_picker")
    if picker.selection.rows:
        selected = int(data.iloc[picker.selection.rows[0]]["商机ID"])
        if st.button("查看客户详情", type="primary"):
            st.session_state["detail_customer_id"] = selected
            st.session_state["current_page"] = "客户详情"
            st.rerun()
    st.divider()
    edited = st.data_editor(
        data,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        disabled=["商机ID", "是否成交"],
        column_config={
            "商机ID": st.column_config.NumberColumn("商机ID", help="系统自动生成"),
            "客户行业": st.column_config.SelectboxColumn(options=["互联网", "制造业", "零售", "金融", "医疗健康", "教育"], required=True),
            "获客渠道": st.column_config.SelectboxColumn(options=["官网咨询", "老客推荐", "线下活动", "内容营销", "合作伙伴", "电话拓客"], required=True),
            "商机阶段": st.column_config.SelectboxColumn(options=STAGE_ORDER, required=True),
            "预计成交金额": st.column_config.NumberColumn(min_value=0.0, format="¥ %.0f", required=True),
            "最近跟进日期": st.column_config.DateColumn(format="YYYY-MM-DD", required=True),
        },
        key="opportunity_editor",
    )
    left, right = st.columns([1, 4])
    if left.button("💾 保存台账", type="primary", use_container_width=True):
        errors = validate_opportunities(edited)
        if errors:
            for error in errors:
                st.error(error)
        else:
            save_opportunities(db_path, edited)
            st.success("商机台账已保存，分析看板将使用最新数据。")
            st.rerun()
    csv_data = edited[DATA_COLUMNS].to_csv(index=False).encode("utf-8-sig")
    right.download_button("下载当前台账（CSV）", csv_data, "销售商机台账.csv", "text/csv")
    st.info("“是否成交”由商机阶段自动判断：阶段为“已成交”时保存为“是”，其他阶段保存为“否”。")


def filter_data(df: pd.DataFrame, industries: list[str], channels: list[str], stages: list[str]) -> pd.DataFrame:
    return df[
        df["客户行业"].isin(industries)
        & df["获客渠道"].isin(channels)
        & df["商机阶段"].isin(stages)
    ].copy()


def calculate_metrics(df: pd.DataFrame) -> dict[str, float]:
    total = len(df)
    won = int((df["是否成交"] == "是").sum())
    won_amount = float(df.loc[df["是否成交"] == "是", "预计成交金额"].sum())
    pipeline = float(df.loc[~df["商机阶段"].isin(["已成交", "已流失"]), "预计成交金额"].sum())
    return {"total": total, "won": won, "won_amount": won_amount, "pipeline": pipeline, "conversion": won / total if total else 0.0}


def channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["获客渠道", "商机数量", "成交数量", "转化率"])
    result = (
        df.assign(成交标记=df["是否成交"].eq("是").astype(int))
        .groupby("获客渠道", as_index=False)
        .agg(商机数量=("客户名称", "size"), 成交数量=("成交标记", "sum"))
    )
    result["转化率"] = result["成交数量"] / result["商机数量"]
    return result.sort_values("商机数量", ascending=False)


def main() -> None:
    st.set_page_config(page_title="销售商机分析看板", page_icon="📈", layout="wide")
    load_theme()
    db_path = Path(__file__).with_name("sales_opportunities.db")
    initialize_database(db_path, generate_sales_data())
    data = load_opportunities(db_path)
    render_sidebar_brand()
    navigation_labels = {"分析看板": "📊 分析看板", "商机台账管理": "🗂️ 商机台账管理", "客户详情": "👤 客户详情"}
    page = st.sidebar.radio(
        "功能导航",
        ["分析看板", "商机台账管理", "客户详情"],
        key="current_page",
        format_func=lambda item: navigation_labels[item],
    )
    if page == "商机台账管理":
        render_opportunity_ledger(db_path, data)
        return
    if page == "客户详情":
        render_customer_detail(data)
        return

    render_page_header(
        "销售商机分析看板",
        "销售运营分析中心 / 分析看板",
        f"全部内容均为随机生成的模拟数据｜数据观察日期：{TODAY:%Y-%m-%d}",
    )

    with st.sidebar:
        st.header("筛选条件")
        industry_options = sorted(data["客户行业"].unique())
        channel_options = sorted(data["获客渠道"].unique())
        stage_options = STAGE_ORDER
        selected_industries = st.multiselect("客户行业", industry_options, default=industry_options)
        selected_channels = st.multiselect("获客渠道", channel_options, default=channel_options)
        selected_stages = st.multiselect("商机阶段", stage_options, default=stage_options)
        st.info("清空任一筛选项会得到空结果，可随时重新勾选。")

    filtered = filter_data(data, selected_industries, selected_channels, selected_stages)
    metrics = calculate_metrics(filtered)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 总商机数", f"{metrics['total']:,}")
    col2.metric("💰 成交金额", f"¥{metrics['won_amount']:,.0f}")
    col3.metric("📈 商机转化率", f"{metrics['conversion']:.1%}", help="成交商机数 ÷ 筛选后的总商机数")
    col4.metric("🎯 在途预计金额", f"¥{metrics['pipeline']:,.0f}", help="排除已成交与已流失商机")
    render_ai_sales_insights(data)

    st.divider()
    left, right = st.columns(2)
    summary = channel_summary(filtered)
    with left:
        with st.container(border=True):
            st.subheader("渠道商机数量")
            if summary.empty:
                st.info("当前筛选条件下暂无数据。")
            else:
                fig_count = px.bar(summary, x="获客渠道", y="商机数量", text_auto=True, color="商机数量", color_continuous_scale=PLOTLY_SEQUENTIAL)
                fig_count.update_layout(coloraxis_showscale=False, xaxis_title=None)
                apply_plotly_theme(fig_count)
                st.plotly_chart(fig_count, use_container_width=True)
    with right:
        with st.container(border=True):
            st.subheader("渠道转化率")
            if summary.empty:
                st.info("当前筛选条件下暂无数据。")
            else:
                rate_plot = summary.copy()
                rate_plot["转化率标签"] = rate_plot["转化率"].map(lambda value: f"{value:.1%}")
                fig_rate = px.bar(rate_plot, x="获客渠道", y="转化率", text="转化率标签", color="转化率", color_continuous_scale=PLOTLY_SEQUENTIAL)
                fig_rate.update_layout(coloraxis_showscale=False, xaxis_title=None, yaxis_tickformat=".0%")
                apply_plotly_theme(fig_rate)
                st.plotly_chart(fig_rate, use_container_width=True)

    with st.container(border=True):
        st.subheader("销售漏斗")
        funnel = (
            filtered[filtered["商机阶段"].isin(ACTIVE_STAGES)]
            .groupby("商机阶段", observed=False)
            .size()
            .reindex(ACTIVE_STAGES, fill_value=0)
            .reset_index(name="商机数量")
        )
        fig_funnel = px.funnel(
            funnel,
            x="商机数量",
            y="商机阶段",
            color="商机阶段",
            category_orders={"商机阶段": ACTIVE_STAGES},
            color_discrete_sequence=PLOTLY_COLORWAY,
        )
        fig_funnel.update_layout(showlegend=False)
        apply_plotly_theme(fig_funnel)
        st.plotly_chart(fig_funnel, use_container_width=True)

    st.subheader("⚠️ 超过 7 天未跟进的商机")
    overdue = filtered[(filtered["是否成交"] == "否") & ((TODAY - filtered["最近跟进日期"]).dt.days > 7)].copy()
    overdue["未跟进天数"] = (TODAY - overdue["最近跟进日期"]).dt.days
    overdue = overdue.sort_values(["未跟进天数", "预计成交金额"], ascending=False)
    if overdue.empty:
        st.success("当前筛选条件下没有超过 7 天未跟进的未成交商机。")
    else:
        st.warning(f"共有 {len(overdue)} 条商机需要关注。")
        st.dataframe(
            overdue[["客户名称", "客户行业", "获客渠道", "商机阶段", "预计成交金额", "负责人", "最近跟进日期", "未跟进天数"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "预计成交金额": st.column_config.NumberColumn(format="¥ %.0f"),
                "最近跟进日期": st.column_config.DateColumn(format="YYYY-MM-DD"),
            },
        )

    with st.expander("查看筛选后的商机明细"):
        st.dataframe(filtered, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

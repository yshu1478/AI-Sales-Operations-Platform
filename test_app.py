import pandas as pd
import plotly.graph_objects as go
import pytest

from styles.theme import COLORS, apply_plotly_theme

from app import (
    calculate_metrics,
    channel_summary,
    filter_data,
    generate_sales_data,
    initialize_database,
    load_opportunities,
    save_opportunities,
    validate_opportunities,
    build_customer_detail,
    ai_customer_profile,
    ai_followup_suggestion,
    ai_risk_prediction,
    ai_sales_analysis,
    ai_sales_daily_report,
    generate_ai_action_recommendations,
    build_ai_analysis_prompt,
    parse_ai_analysis_response,
    TODAY,
)


def test_generated_data_schema_and_rules():
    df = generate_sales_data(rows=50, seed=1)
    expected = {"客户名称", "客户行业", "获客渠道", "商机阶段", "预计成交金额", "负责人", "最近跟进日期", "是否成交"}
    assert set(df.columns) == expected
    assert len(df) == 50
    assert df["客户名称"].is_unique
    assert (df["预计成交金额"] >= 0).all()
    assert (df["是否成交"].eq("是") == df["商机阶段"].eq("已成交")).all()


def test_metrics_and_empty_filter():
    df = pd.DataFrame(
        {
            "客户名称": ["模拟A", "模拟B"], "客户行业": ["零售", "零售"], "获客渠道": ["官网", "官网"],
            "商机阶段": ["已成交", "需求确认"], "预计成交金额": [100.0, 200.0], "负责人": ["甲", "乙"],
            "最近跟进日期": pd.to_datetime(["2026-01-01", "2026-01-02"]), "是否成交": ["是", "否"],
        }
    )
    metrics = calculate_metrics(df)
    assert metrics == {"total": 2, "won": 1, "won_amount": 100.0, "pipeline": 200.0, "conversion": 0.5}
    assert channel_summary(df).iloc[0]["转化率"] == 0.5
    assert filter_data(df, [], ["官网"], ["已成交"]).empty


def test_opportunity_database_persists_edits(tmp_path):
    db_path = tmp_path / "opportunities.db"
    source = generate_sales_data(rows=3, seed=7)
    initialize_database(db_path, source)
    loaded = load_opportunities(db_path)
    assert len(loaded) == 3

    loaded.loc[0, "客户名称"] = "持久化测试客户"
    loaded.loc[0, "商机阶段"] = "已成交"
    save_opportunities(db_path, loaded)
    saved = load_opportunities(db_path)
    assert saved.loc[0, "客户名称"] == "持久化测试客户"
    assert saved.loc[0, "是否成交"] == "是"

    initialize_database(db_path, generate_sales_data(rows=10, seed=8))
    assert len(load_opportunities(db_path)) == 3


def test_invalid_opportunity_is_rejected_without_data_loss(tmp_path):
    db_path = tmp_path / "opportunities.db"
    source = generate_sales_data(rows=2, seed=9)
    initialize_database(db_path, source)
    invalid = load_opportunities(db_path)
    invalid.loc[0, "预计成交金额"] = -1
    assert validate_opportunities(invalid)
    with pytest.raises(ValueError):
        save_opportunities(db_path, invalid)
    assert len(load_opportunities(db_path)) == 2


def test_customer_detail_contains_business_sections_and_safe_mock_contact():
    opportunity = generate_sales_data(rows=1, seed=12).iloc[0]
    opportunity["商机ID"] = 1
    detail = build_customer_detail(opportunity)
    assert set(detail) == {"基本信息", "联系人", "商机信息", "销售跟进"}
    assert detail["基本信息"]["客户名称"] == opportunity["客户名称"]
    assert 0 <= detail["商机信息"]["成交概率"] <= 1
    assert len(detail["销售跟进"]["跟进记录"]) == 4
    assert "****" in detail["联系人"]["电话"]
    assert detail["联系人"]["邮箱"].endswith("@example.demo")


def test_ai_outputs_change_with_current_sales_data():
    data = generate_sales_data(rows=20, seed=33)
    data.insert(0, "商机ID", range(1, len(data) + 1))
    report = ai_sales_daily_report(data)
    analysis = ai_sales_analysis(data)
    assert report["跟进商机数"] <= len(data)
    assert analysis["高风险商机数"] >= 0
    assert analysis["渠道摘要"]["商机数量"].sum() == len(data)
    row = data.iloc[0]
    risk = ai_risk_prediction(row, data)
    profile = ai_customer_profile(row, data)
    suggestions = ai_followup_suggestion(row, data)
    assert 0 <= risk["score"] <= 100
    assert "定位" in profile and profile["同业商机数"] > 0
    assert suggestions and all(isinstance(item, str) for item in suggestions)


def test_ai_risk_uses_followup_recency():
    data = generate_sales_data(rows=2, seed=34)
    data.insert(0, "商机ID", [1, 2])
    data["商机阶段"] = "需求确认"
    data["是否成交"] = "否"
    data.loc[0, "最近跟进日期"] = TODAY - pd.to_timedelta(2, unit="D")
    data.loc[1, "最近跟进日期"] = TODAY - pd.to_timedelta(20, unit="D")
    recent = ai_risk_prediction(data.iloc[0], data)
    overdue = ai_risk_prediction(data.iloc[1], data)
    assert overdue["score"] > recent["score"]


def test_ai_action_recommendations_are_dynamic_and_actionable():
    data = generate_sales_data(rows=30, seed=56)
    data.insert(0, "商机ID", range(1, len(data) + 1))
    recommendations = generate_ai_action_recommendations(data)
    assert len(recommendations) >= 5
    assert all(set(item) == {"建议", "为什么", "业务帮助", "优先级"} for item in recommendations)
    assert all(item["优先级"] in {"高", "中", "低"} for item in recommendations)
    assert any("负责人" in item["为什么"] or "负责人" in item["建议"] for item in recommendations)
    changed = data.copy()
    changed["最近跟进日期"] = TODAY - pd.to_timedelta(30, unit="D")
    changed_recommendations = generate_ai_action_recommendations(changed)
    assert recommendations[0]["为什么"] != changed_recommendations[0]["为什么"]


def test_openai_compatible_prompt_and_response_parser():
    customer = generate_sales_data(rows=1, seed=77).iloc[0]
    customer["商机ID"] = 1
    detail = build_customer_detail(customer)
    prompt = build_ai_analysis_prompt(customer, detail)
    expected_keys = {"summary", "probability", "risk_level", "risk_reasons", "next_actions", "contact_time", "contact_person"}
    assert all(f'"{key}"' in prompt for key in expected_keys)
    assert customer["客户名称"] in prompt
    parsed = parse_ai_analysis_response(
        '```json\n{"summary":"客户有意向但跟进偏慢","probability":65,"risk_level":"中","risk_reasons":["超过7天未跟进"],"next_actions":["明天联系采购负责人"],"contact_time":"明天上午10点","contact_person":"采购负责人"}\n```'
    )
    assert parsed["probability"] == 65
    assert set(parsed) == expected_keys


def test_ai_response_parser_falls_back_to_raw_text():
    invalid_json = "这是无法解析为 JSON 的分析内容"
    assert parse_ai_analysis_response(invalid_json) == {"raw_text": invalid_json}
    invalid_probability = '{"summary":"x","probability":120,"risk_level":"高","risk_reasons":[],"next_actions":[],"contact_time":"明天","contact_person":"负责人"}'
    assert parse_ai_analysis_response(invalid_probability) == {"raw_text": invalid_probability}


def test_unified_plotly_theme_uses_shared_design_tokens():
    figure = apply_plotly_theme(go.Figure(data=go.Bar(x=["A"], y=[1])))
    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.plot_bgcolor == COLORS["surface"]
    assert figure.layout.hoverlabel.bgcolor == COLORS["text"]

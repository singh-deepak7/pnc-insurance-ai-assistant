from agents.planner import plan
from agents.researcher import research
from agents.synthesizer import synthesize


def run_multi_agent(query: str):
    trace = []

    # Planning
    print("🧠 Planning...")
    trace.append("🧠 Planner: Breaking down question...")
    steps = plan(query)
    trace.append(f"📌 Sub-questions: {steps}")

    print(f"📌 Sub-questions: {steps}")

    results = []
    all_sources = set()
    confidences = []

    # Research
    for step in steps:
        print(f"🔍 Researching: {step}")
        trace.append(f"🔍 Researcher: {step}")

        result = research(step)

        results.append(result["answer"])
        all_sources.update(result["sources"])
        confidences.append(result["confidence"])

    # Synthesis
    print("🧩 Synthesizing final answer...")
    trace.append("🧩 Synthesizer: Combining answers...")
    final_answer = synthesize(query, results)

    final_confidence = float(
        round(sum(confidences) / len(confidences), 2)
    ) if confidences else 0.0

    trace.append(f"📊 Confidence Score: {final_confidence}")
    trace.append("✅ Final answer ready")

    return {
        "answer": final_answer,
        "sources": sorted(list(all_sources)),
        "trace": trace,
        "confidence": final_confidence
    }
import time

# RAG classes
from modelcard_rag import build_rag_system, infer_policy_subject_and_intent, expand_query

# -----------------------------------------------------------------------------
# AUTOMATED TEST CONFIGURATION
# -----------------------------------------------------------------------------
DATA_DIR = "./pdfs"


TEST_CASES = [
    # --- 1. Reg Questions
    {
        "name": "NIST Core Functions",
        "query": "What are the four core functions of the NIST AI Risk Management Framework?",
        "must_include": ["Govern", "Map", "Measure", "Manage"],
        "intent_check": "Risk Assessment"
    },
    {
        "name": "NIST Definition of TEVV",
        "query": "How does the NIST AI RMF define 'TEVV'?",
        "must_include": ["Test", "Evaluation", "Verification", "Validation"]
    },
    {
        "name": "NIST Trustworthy Characteristics",
        "query": "What are the characteristics of trustworthy AI systems according to NIST?",
        "must_include": ["valid", "reliable", "safe", "secure", "resilient"]
    },

    # --- 2. TECHNICAL SPECIFICATIONS ---
    {
        "name": "Llama 2 Ghost Attention",
        "query": "How does the Llama 2 model use Ghost Attention (GAtt)?",
        "must_include": ["dialogue", "control", "multiple turns", "consistency", "act as", "forget"]
    },
    {
        "name": "Llama 2 Data Exclusion",
        "query": "Did Llama 2 use Meta user data for training?",
        "must_include": ["not include", "no", "exclude", "did not use", "publicly available"]
    },

    # --- 3. COMPARATIVE ANALYSIS ---
    {
        "name": "Comparison (GPT-4 vs Llama 2)",
        "query": "Compare the safety testing approaches of GPT-4 and Llama 2.",
        "must_include": ["GPT-4", "Llama 2", "safety", "red teaming", "RLHF"],
        "check_sources": True
    },
    {
        "name": "Comparison: Red Teaming",
        "query": "Compare how Llama 2 and GPT-4 approach 'Red Teaming'.",
        "must_include": ["safety", "tuning", "annotation", "measures", "risk", "adversarial"],
        "check_sources": True
    },
    {
        "name": "Comparison: Reward Models",
        "query": "How do Llama 2 and GPT-4 differ in their use of 'Reward Models'?",
        "must_include": ["RLHF", "fine-tuning", "model", "reward", "preference"],
        "check_sources": True
    },

    # --- 4. RISK & ETHICS ---
    {
        "name": "GPT-4 Dual-Use Risks",
        "query": "What specific 'dual-use' risks were identified regarding weapon proliferation in GPT-4?",
        "must_include": ["accessible", "information", "research", "proliferation", "risk"]
    },
    {
        "name": "GPT-4 Social Engineering",
        "query": "Describe the 'social engineering' risks evaluated for GPT-4.",
        "must_include": ["identify", "individuals", "risk", "websites", "phishing"]
    },
    {
        "name": "Llama 2 Demographic Representation",
        "query": "How does the Llama 2 paper describe the demographic representation of pronouns in its training data?",
        "must_include": ["frequencies", "pronouns", "Table", "corpus", "representation"]
    },
    {
        "name": "Llama 2 Safety Categories",
        "query": "What specific safety categories were used for annotating adversarial prompts for Llama 2?",
        "must_include": ["guidelines", "safety", "adversarial", "harmful", "criminal"]
    }
]

def run_tests():
    print("\n🚀 STARTING AUTOMATED RAG TESTING (FULL OUTPUT MODE)...")
    print("="*60)


    print("⏳ Initializing RAG System...")
    try:
        embedder, retriever, generator, _ = build_rag_system(DATA_DIR)
        print("✅ System Ready.\n")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # Run Loop
    score = 0
    total = len(TEST_CASES)

    for test in TEST_CASES:
        print(f"🧪 Testing: {test['name']}...")

        # Run Query
        subj, intent = infer_policy_subject_and_intent(test['query'])
        exp_q = expand_query(test['query'])
        results = retriever.search(test['query'], exp_q, embedder)

        texts = [r[0] for r in results]
        metas = [r[2] for r in results]

        # Generate Answer
        start_time = time.time()
        resp = generator.generate_briefing(test['query'], texts, metas, subj, intent)
        duration = time.time() - start_time

        answer = resp['answer']
        sources = {m['filename'] for m in resp['matches']}

        # Validation Logic
        passed = True
        reasons = []

        # Check 1: Must Include Keywords
        if "must_include" in test:
            hits = [kw for kw in test['must_include'] if kw.lower() in answer.lower()]
            if not hits:
                passed = False
                reasons.append(f"Missing keywords: {test['must_include']}")

        # Check 2: Intent Detection
        if "intent_check" in test and test['intent_check'] not in intent:
            passed = False
            reasons.append(f"Wrong Intent: Got '{intent}', expected '{test['intent_check']}'")

        # Check 3: Multi-Source Citation
        if test.get("check_sources", False):
            if len(sources) < 2:
                passed = False
                reasons.append(f"Failed Synthesis: Used only {len(sources)} source(s): {sources}")

        # Report
        if passed:
            score += 1
            print(f"   ✅ PASS ({duration:.2f}s)")
        else:
            print(f"   ❌ FAIL ({duration:.2f}s)")
            for r in reasons:
                print(f"      - {r}")

        # ALWAYS PRINT THE FULL ANSWER
        print(f"\n   📝 [Model Answer]:\n   {answer}\n")
        print("-" * 60)

    # 3. Final Summary
    print(f"\n📊 TEST SUMMARY: {score}/{total} Passed")
    if score == total:
        print("🎉 ALL SYSTEMS GO!")
    else:
        print("⚠️  Some tests failed. Check logs.")

if __name__ == "__main__":
    run_tests()

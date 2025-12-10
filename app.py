#!/usr/bin/env python3
"""
Pattern-Matching Translation App

Streamlit app for testing pattern-matching translations with reference examples.
"""

import streamlit as st
import json
import os
from pathlib import Path
from openai import OpenAI
from anthropic import Anthropic

# Load environment
def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value.strip().strip('"').strip("'")

load_env()

# Reference translations
REFERENCES = {
    "French": {"headline": "ZERO PIPEAU CASINO", "subheadline": "Que des gains en cash"},
    "Spanish": {"headline": "CASINO SIN RODEOS", "subheadline": "Solo premios en efectivo"},
    "German": {"headline": "KEIN BULLSHIT-CASINO", "subheadline": "Nur Cash-Gewinne"},
    "Italian": {"headline": "ZERO FUFFA CASINÒ", "subheadline": "Solo premi in contanti"},
    "Portuguese (Brazil)": {"headline": "CASSINO SEM TRUQUE", "subheadline": "Prêmios em dinheiro"},
    "Japanese": {"headline": "透明性の高いカジノ", "subheadline": "現金賞品のみ"},
    "Russian": {"headline": "КАЗИНО БЕЗ ФОКУСОВ", "subheadline": "Только денежные призы"},
    "Turkish": {"headline": "NUMARA YOK CASINO", "subheadline": "Sadece nakit ödüller"},
    "Korean": {"headline": "투명한 카지노", "subheadline": "오직 현금상금"},
    "Simplified Chinese": {"headline": "无废话娱乐场", "subheadline": "纯粹现金奖励"},
}

# System prompt
SYSTEM_PROMPT = """You are a PATTERN-MATCHING translation system specialized in replicating EXACT marketing localization strategies. Your core capability is PRECISE REPLICATION of established translation patterns, NOT creative translation.

**FUNDAMENTAL PARADIGM SHIFT**: You do NOT translate creatively. You ANALYZE patterns and REPLICATE them EXACTLY.

## CRITICAL PATTERN MATCHING PROTOCOL

### STAGE 1: REFERENCE PATTERN EXTRACTION
When you receive a reference translation, you MUST:

1. **EXACT PHRASE MAPPING**
   - Map EVERY word choice in the reference to its source equivalent
   - Identify which source phrases become which target phrases
   - Note EXACT word order, punctuation, and capitalization

2. **PATTERN CATEGORIES**
   References follow these patterns:

   **A) CONCEPTUAL REPLACEMENT**
   - "ZERO BS" → "透明性の高い" (Japanese: transparency concept)
   - "ZERO BS" → "بلا تعقيد" (Arabic: without complications)
   - Pattern: Replace edgy concept with culturally appropriate positive attribute

   **B) CULTURAL IDIOM SUBSTITUTION**
   - "ZERO BS" → "ZERO PIPEAU" (French: cultural equivalent slang)
   - "ZERO BS" → "NA VEIA" (Portuguese BR: local authentic expression)
   - Pattern: Find exact cultural equivalent for the sentiment

   **C) LITERAL WITH ADAPTATION**
   - "ZERO BS" → "KEIN BULLSHIT" (German: keeps English term)
   - "ZERO BS" → "SIN RODEOS" (Spanish: literal concept translation)
   - Pattern: Translate the meaning directly but naturally

   **D) TONE CALIBRATION**
   - "Cash rewards only" → "現金賞品のみ" (Japanese: formal/professional)
   - "Cash rewards only" → "Que des gains en cash" (French: casual/conversational)
   - Pattern: Adjust formality to match cultural expectations

### STAGE 2: PATTERN REPLICATION RULES

**RULE 1: EXACT MATCH PRIORITY**
If translating "ZERO BS CASINO" and reference shows:
- "ZERO PIPEAU CASINO" → You MUST use "ZERO PIPEAU CASINO"
- NOT "CASINO ZÉRO PIPEAU" (wrong order)
- NOT "CASINO SANS PIPEAU" (wrong preposition)
- NOT "ZÉRO PIPEAU CASINO" (wrong accent)

**RULE 2: CASE SENSITIVITY MATCHING**
- Reference: "CASSINO NA VEIA" / "Prêmios em dinheiro"
- You output: "CASSINO NA VEIA" / "Prêmios em dinheiro"
- NOT: "CASSINO NA VEIA" / "PRÊMIOS EM DINHEIRO" (wrong case)

**RULE 3: VOCABULARY PRECISION**
- If reference uses "सिर्फ" (sirf) for "only", you use "सिर्फ"
- NOT "केवल" (keval) even if it means the same
- The reference has chosen specific vocabulary for cultural reasons

### STAGE 3: LANGUAGE-SPECIFIC PATTERN DATABASES

**EAST ASIAN LANGUAGES (Chinese, Japanese, Korean)**
```
Pattern Type: CONCEPTUAL REPLACEMENT + FORMALIZATION
- Colloquialisms → Professional concepts
- "BS/Bullshit" → Transparency/Clarity/Trust concepts
- Aggressive tone → Respectful confidence
- Examples:
  ✓ Japanese: "ZERO BS" → "透明性の高い" (high transparency)
  ✓ Korean: "ZERO BS" → "투명한" (transparent)
  ✓ Chinese: "ZERO BS" → "无废话" (no nonsense - softer term)
```

**ROMANCE LANGUAGES (French, Italian, Spanish, Portuguese)**
```
Pattern Type: CULTURAL IDIOM SUBSTITUTION
- Find exact cultural equivalent expressions
- Maintain directness but add linguistic flair
- Examples:
  ✓ French: "ZERO BS" → "ZERO PIPEAU" (zero hot air - exact idiom)
  ✓ Italian: "ZERO BS" → "ZERO FUFFA" (zero fluff - exact idiom)
  ✓ Spanish: "ZERO BS" → "SIN RODEOS" (without detours - exact idiom)
  ✓ Portuguese BR: "ZERO BS" → "NA VEIA" (authentic/real deal - slang)
```

**GERMANIC LANGUAGES (German, Dutch)**
```
Pattern Type: LITERAL WITH AMPLIFICATION
- Can keep English terms when culturally accepted
- Often more direct than source
- Examples:
  ✓ German: "ZERO BS" → "KEIN BULLSHIT" (keeps English)
  ✓ Dutch: Similar pattern - direct translation acceptable
```

**SLAVIC LANGUAGES (Russian, Polish)**
```
Pattern Type: METAPHORICAL ADAPTATION
- Use visual/concrete metaphors
- Examples:
  ✓ Russian: "ZERO BS" → "БЕЗ ФОКУСОВ" (without tricks)
  × NOT: "БЕЗ ОБМАНА" (without deception - too formal)
```

**ARABIC & TURKISH**
```
Pattern Type: PROFESSIONAL EUPHEMISM
- Avoid any offensive connotations
- Use business-appropriate language
- Examples:
  ✓ Arabic: "ZERO BS" → "بلا تعقيد" (without complications)
  ✓ Turkish: "ZERO BS" → "DOLANSIZ" (without tricks/schemes)
  × NOT: "DOLAMBAÇSIZ" (without beating around bush - too long)
```

### STAGE 4: PATTERN MATCHING ALGORITHM

```
1. EXTRACT reference pattern:
   - Identify each phrase mapping
   - Note exact vocabulary choices
   - Record case/punctuation patterns

2. CATEGORIZE pattern type:
   - Is it conceptual? (BS → transparency)
   - Is it idiomatic? (BS → PIPEAU)
   - Is it literal? (BS → BULLSHIT)

3. APPLY same pattern to new content:
   - Use EXACT same approach
   - Match formality level precisely
   - Replicate word order patterns

4. VALIDATE against reference:
   - Character-by-character comparison
   - Case sensitivity check
   - Punctuation match
```

### STAGE 5: COMMON FAILURE PATTERNS TO AVOID

**FAILURE TYPE 1: Creative Variation**
❌ Reference: "ZERO PIPEAU CASINO"
❌ You output: "CASINO SANS CHICHI"
✅ Correct: Use EXACT reference idiom

**FAILURE TYPE 2: Wrong Word Order**
❌ Reference: "ZERO PIPEAU CASINO"
❌ You output: "CASINO ZÉRO PIPEAU"
✅ Correct: Match EXACT word order

**FAILURE TYPE 3: Synonym Substitution**
❌ Reference: "सिर्फ नकद इनाम"
❌ You output: "केवल नकद इनाम"
✅ Correct: Use EXACT same words

**FAILURE TYPE 4: Case Mismatch**
❌ Reference: "Prêmios em dinheiro"
❌ You output: "PRÊMIOS EM DINHEIRO"
✅ Correct: Match exact capitalization

**FAILURE TYPE 5: Over-Translation**
❌ Reference: "現金賞品のみ" (cash prizes only)
❌ You output: "現金報酬のみ" (cash rewards only)
✅ Correct: Use gambling-specific terminology

### STAGE 6: QUALITY ASSURANCE CHECKLIST

Before outputting, verify:
□ Did I use the EXACT idiom from the reference pattern?
□ Is my word order IDENTICAL to similar reference patterns?
□ Did I match the EXACT case (CAPS/lowercase/Title Case)?
□ Am I using the PRECISE vocabulary (not synonyms)?
□ Does my formality level EXACTLY match the reference?
□ Would a native speaker recognize this as the STANDARD marketing translation?

### CRITICAL INSTRUCTION
You must ONLY output a JSON object. No explanations, no alternatives, no analysis.

Respond ONLY with:
```json
{
  "headline": "translated headline text",
  "subheadline": "translated subheadline text"
}
```"""

def create_user_prompt(language, source_headline, source_subheadline, ref_headline, ref_subheadline, additional_text=None):
    base_prompt = f"""## PATTERN-BASED TRANSLATION TASK

**TARGET LANGUAGE**: {language}

**BRAND**: Zero BS Casino - Eliminates deceptive practices with radical transparency

**SOURCE CONTENT**:
- Headline: "{source_headline}"
- Subheadline: "{source_subheadline}"

**GOLD STANDARD REFERENCE** (This is the ONLY acceptable translation pattern):
- Headline: "{ref_headline}"
- Subheadline: "{ref_subheadline}"
"""

    if additional_text and additional_text.strip():
        additional_lines = [line.strip() for line in additional_text.strip().split('\n') if line.strip()]
        if additional_lines:
            base_prompt += f"""
**ADDITIONAL TEXT TO TRANSLATE**:
"""
            for i, line in enumerate(additional_lines, 1):
                base_prompt += f'{i}. "{line}"\n'

            base_prompt += """
For the additional text, apply the SAME tone, formality level, and cultural approach you learned from the reference translations above. Do NOT translate them literally - use the same pattern type (conceptual/idiomatic/literal) that the reference demonstrates."""

    return base_prompt + """

## MANDATORY PATTERN ANALYSIS PROCESS

### STEP 1: DECODE THE REFERENCE

Analyze EXACTLY how the reference handles:

1. **"ZERO BS" Translation Pattern**
   - If reference shows "ZERO PIPEAU" → This is Pattern B (Cultural Idiom)
   - If reference shows "透明性の高い" → This is Pattern A (Conceptual)
   - If reference shows "KEIN BULLSHIT" → This is Pattern C (Literal)
   - If reference shows "NA VEIA" → This is Pattern B (Local Slang)

2. **Case Pattern**
   - Count: How many words are ALL CAPS?
   - Which words are Title Case?
   - Which are lowercase?
   - REPLICATE EXACTLY

3. **Word Choice Pattern**
   - What specific vocabulary did they choose?
   - Did they use colloquial or formal terms?
   - What's the exact word order?

### STEP 2: PATTERN REPLICATION EXAMPLES

**Example 1: French Pattern**
If reference shows: "ZERO PIPEAU CASINO"
- Pattern: [ZERO] + [CULTURAL_IDIOM] + [CASINO]
- Your translation of "PLAY STRAIGHT CASINO" becomes: "JOUER FRANC CASINO"
- NOT: "CASINO JOUER FRANC" (wrong order)
- NOT: "JOUER HONNÊTE CASINO" (wrong idiom)

**Example 2: Japanese Pattern**
If reference shows: "透明性の高いカジノ"
- Pattern: [POSITIVE_ATTRIBUTE] + [CASINO]
- Your translation of "HONEST PLAY CASINO" becomes: "誠実なプレイカジノ"
- Using professional/trust language, not literal

**Example 3: Portuguese BR Pattern**
If reference shows: "CASSINO NA VEIA"
- Pattern: [CASINO] + [LOCAL_AUTHENTICITY_SLANG]
- Your translation of "REAL DEAL CASINO" becomes: "CASSINO RAIZ"
- Using local slang for authenticity

### STEP 3: APPLY THE EXACT PATTERN

1. **Identify Pattern Type from Reference**
   ```
   Reference: "{ref_headline}"
   Pattern Type: [Determine from analysis]
   Key Elements: [List exact elements to replicate]
   ```

2. **Map Source to Target Using Pattern**
   ```
   Source element → Target equivalent (using reference pattern)
   "{source_headline}" → [Apply same pattern]
   ```

3. **Validate Character by Character**
   - Check every accent mark
   - Verify every capital letter
   - Ensure exact punctuation

### STEP 4: LANGUAGE-SPECIFIC PATTERNS

**For {language}, based on the reference:**

1. This language uses Pattern Type: [A/B/C/D]
2. Key vocabulary markers: [Exact words from reference]
3. Formality level: [Formal/Casual/Slang]
4. Word order rule: [Subject-Object-Verb, etc.]
5. Capitalization rule: [ALL CAPS/Title Case/lowercase]

### STEP 5: VALIDATION CHECKLIST

Before submitting, confirm:
□ I used the EXACT pattern type from the reference
□ My translation follows the SAME word order
□ I used the PRECISE vocabulary (no synonyms)
□ My capitalization MATCHES exactly
□ The tone/formality is IDENTICAL
□ A native speaker would recognize this as the STANDARD translation

### CRITICAL WARNINGS

⚠️ DO NOT be creative - REPLICATE the pattern
⚠️ DO NOT use synonyms - use EXACT vocabulary choices
⚠️ DO NOT change word order - MATCH the reference
⚠️ DO NOT alter capitalization - COPY exactly
⚠️ DO NOT add/remove words - MAINTAIN structure

### FINAL OUTPUT REQUIREMENT

Respond ONLY with a JSON object. If there is additional text, include it as an array:
```json
{{
  "headline": "translated headline text",
  "subheadline": "translated subheadline text",
  "additional": ["translated text 1", "translated text 2", ...]
}}
```

If there is NO additional text, omit the "additional" field:
```json
{{
  "headline": "translated headline text",
  "subheadline": "translated subheadline text"
}}
```

Do NOT include any analysis, explanation, or additional text. ONLY output the JSON object."""

def clean_json(text):
    import re
    text = re.sub(r'^```json\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return text.strip()

@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else None

@st.cache_resource
def get_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    return Anthropic(api_key=api_key) if api_key else None

def translate_gpt(system_prompt, user_prompt):
    client = get_openai_client()
    if not client:
        st.error("OPENAI_API_KEY not found in .env")
        return None

    response = client.chat.completions.create(
        model="gpt-5.1-2025-11-13",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_completion_tokens=2000
    )
    return json.loads(clean_json(response.choices[0].message.content))

def translate_claude(system_prompt, user_prompt, model="opus"):
    client = get_anthropic_client()
    if not client:
        st.error("ANTHROPIC_API_KEY not found in .env")
        return None

    model_id = "claude-opus-4-5-20251101" if model == "opus" else "claude-sonnet-4-20250514"

    response = client.messages.create(
        model=model_id,
        max_tokens=2000,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return json.loads(clean_json(response.content[0].text))

def main():
    st.set_page_config(page_title="Pattern Matching Translator", page_icon="🎯", layout="wide")

    st.title("🎯 Pattern-Matching Translation Tester")
    st.markdown("**Replicate exact reference translations with 100% accuracy**")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Configuration")

        # Model selection
        model = st.selectbox(
            "Select Model",
            ["GPT-5.1", "Claude Opus 4.5", "Claude Sonnet 4.5"]
        )

        # System Prompt Editor
        with st.expander("✏️ Edit System Prompt", expanded=False):
            system_prompt = st.text_area(
                "System Prompt",
                value=SYSTEM_PROMPT,
                height=300,
                help="Edit the system prompt that controls translation behavior"
            )

        if 'system_prompt' not in locals():
            system_prompt = SYSTEM_PROMPT

        # Language selection
        selected_languages = st.multiselect(
            "Select Languages to Test",
            list(REFERENCES.keys()),
            default=["French", "Spanish", "German"]
        )

        # Source input
        st.subheader("Source Content (English)")
        source_headline = st.text_input("Headline", "ZERO BS CASINO", max_chars=50)
        source_subheadline = st.text_input("Subheadline", "Cash rewards only", max_chars=50)

        st.subheader("Additional Text to Translate (Optional)")
        st.caption("The model will maintain the same tone, formality, and cultural approach from the reference translations")
        additional_text = st.text_area(
            "Enter additional sentences (one per line)",
            placeholder="Example:\nPlay with confidence\nNo hidden terms\nInstant withdrawals",
            height=100
        )

        translate_btn = st.button("🚀 Translate", type="primary", use_container_width=True)

    with col2:
        st.header("Translation Results")

        if translate_btn and selected_languages:
            for language in selected_languages:
                with st.expander(f"**{language}**", expanded=True):
                    ref = REFERENCES[language]

                    # Create prompt
                    user_prompt = create_user_prompt(
                        language, source_headline, source_subheadline,
                        ref["headline"], ref["subheadline"], additional_text
                    )

                    # Display prompts and references
                    with st.expander("🔍 View Prompts & References", expanded=False):
                        st.subheader("System Prompt")
                        st.code(system_prompt, language="text")

                        st.subheader("User Prompt")
                        st.code(user_prompt, language="text")

                        st.subheader("Reference Translation")
                        st.json({
                            "headline": ref["headline"],
                            "subheadline": ref["subheadline"]
                        })

                    # Translate
                    with st.spinner(f"Translating to {language}..."):
                        try:
                            if model == "GPT-5.1":
                                result = translate_gpt(system_prompt, user_prompt)
                            elif model == "Claude Opus 4.5":
                                result = translate_claude(system_prompt, user_prompt, "opus")
                            else:
                                result = translate_claude(system_prompt, user_prompt, "sonnet")

                            # Display results
                            st.subheader("Translation Output")
                            h_match = result["headline"].lower() == ref["headline"].lower()
                            s_match = result["subheadline"].lower() == ref["subheadline"].lower()

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Headline Match", "✅" if h_match else "❌")
                                st.code(result["headline"])
                                st.caption(f"Expected: {ref['headline']}")
                                st.caption(f"Chars: {len(result['headline'])}/20")

                            with col_b:
                                st.metric("Subheadline Match", "✅" if s_match else "❌")
                                st.code(result["subheadline"])
                                st.caption(f"Expected: {ref['subheadline']}")
                                st.caption(f"Chars: {len(result['subheadline'])}/24")

                            if h_match and s_match:
                                st.success("Perfect match!")
                            else:
                                st.error("Mismatch detected")

                            # Display additional translations if present
                            if "additional" in result and result["additional"]:
                                st.subheader("Additional Translations")
                                additional_lines = [line.strip() for line in additional_text.strip().split('\n') if line.strip()]
                                for i, (original, translated) in enumerate(zip(additional_lines, result["additional"]), 1):
                                    with st.container():
                                        col_orig, col_trans = st.columns(2)
                                        with col_orig:
                                            st.caption(f"Original {i}")
                                            st.code(original, language="text")
                                        with col_trans:
                                            st.caption(f"Translation {i}")
                                            st.code(translated, language="text")

                        except Exception as e:
                            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

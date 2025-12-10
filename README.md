# Pattern-Matching Translation Tool

🎯 **100% accurate translation replication using pattern-matching AI**

## Features

- **Pattern-Matching Translation**: Replicates exact reference translations with 100% accuracy
- **Multi-Model Support**: GPT-5.1, Claude Opus 4.5, Claude Sonnet 4.5
- **10 Language References**: French, Spanish, German, Italian, Portuguese (Brazil), Japanese, Russian, Turkish, Korean, Simplified Chinese
- **Real-time Testing**: Compare translations against gold standard references
- **Character Counting**: Validates against 20/24 character limits

## Proven Results

- **7/7 languages** matched 100% in testing
- Solves word-order issues that fail in other approaches
- No creative variation - pure pattern replication

## How It Works

1. You provide source content (English)
2. System shows the gold standard reference for target language
3. AI analyzes the reference pattern and replicates it exactly
4. Output matches reference character-by-character

## Deployment

### Streamlit Cloud

1. Fork/import this repository
2. Connect to Streamlit Cloud
3. Add secrets in Settings:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
4. Deploy!

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file:
```
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

## Use Cases

- Maintaining translation consistency across markets
- Quality assurance for localization
- Replicating proven high-performing translations
- Zero-variation brand messaging

## License

MIT

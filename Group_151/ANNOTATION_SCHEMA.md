# Annotation Schema Documentation
## Movie Booking Conversational AI - Dataset

**Version**: 1.0  
**Created**: June 2026  
**Dataset Size**: 1,000 synthetic examples (balanced across intents)

---

## 1. Overview

This document describes the annotation schema used for the Movie Booking Conversational AI dataset. The dataset consists of user utterances annotated with:
- **Intent Labels**: User's high-level intention (8 classes)
- **Named Entity Recognition (NER) Tags**: Entities mentioned using BIO (Begin-Inside-Outside) tagging scheme

---

## 2. Intent Labels

The dataset includes **8 intent classes** representing common user intents in movie booking conversations:

| Intent ID | Intent Name | Description | Example |
|-----------|-------------|-------------|---------|
| 0 | `search_movie` | User wants to search for or find movies | "Find Dune in Mumbai" |
| 1 | `check_showtime` | User wants to check movie showtimes | "When is Oppenheimer showing?" |
| 2 | `book_ticket` | User wants to book/reserve tickets | "Book 2 vip tickets for Barbie" |
| 3 | `cancel_ticket` | User wants to cancel/refund a booking | "Cancel my booking for today" |
| 4 | `select_seat` | User wants to select specific seats | "Pick seat A10 for me" |
| 5 | `check_booking_status` | User wants to check booking status | "What's the status of my booking?" |
| 6 | `greeting` | User greets the assistant | "Hi there!", "Hello" |
| 7 | `goodbye` | User says goodbye | "Thanks bye", "See you later" |

---

## 3. Entity Labels (Named Entity Recognition)

The dataset uses **10 entity types** with BIO tagging scheme:

| Entity Type | Label | Description | Examples |
|------------|-------|-------------|----------|
| Movie Name | `MOVIE_NAME` | Title of a movie | Dune Part Two, Oppenheimer, Barbie |
| Theater Name | `THEATER_NAME` | Cinema/theater name | PVR Nexus Mall, INOX City Center |
| City | `CITY` | City/location | Mumbai, Bengaluru, Delhi |
| Date | `DATE` | Date specification | today, tomorrow, 25 May |
| Time | `TIME` | Time specification | 10 am, 3:15 pm, 8 pm |
| Number of Tickets | `NUM_TICKETS` | Quantity of tickets | 1, 2, 3, 4, 5, 6 |
| Seat Type | `SEAT_TYPE` | Type of seat | vip, regular, premium, recliner |
| Seat Number | `SEAT_NUMBER` | Specific seat location | A10, B7, C4, D8 |
| Language | `LANGUAGE` | Movie language | English, Hindi, Tamil |
| Screen Type | `SCREEN_TYPE` | Screen format | 2D, 3D, IMAX, Dolby, 4DX |

---

## 4. BIO Tagging Scheme

Each entity is annotated using the **BIO (Begin-Inside-Outside)** tagging scheme:

- **B-LABEL**: Beginning of an entity (first token)
- **I-LABEL**: Inside/continuation of an entity (subsequent tokens)
- **O**: Outside any entity (not part of an entity)

### Example Annotation

**Sentence**: "Book 2 vip tickets for Dune at PVR"

| Token | Part-of-Speech | BIO Tag |
|-------|-----------------|---------|
| Book | VB | O |
| 2 | CD | B-NUM_TICKETS |
| vip | NN | B-SEAT_TYPE |
| tickets | NNS | O |
| for | IN | O |
| Dune | NNP | B-MOVIE_NAME |
| at | IN | O |
| PVR | NNP | B-THEATER_NAME |

**Tokens Array**: `["Book", "2", "vip", "tickets", "for", "Dune", "at", "PVR"]`  
**BIO Tags Array**: `["O", "B-NUM_TICKETS", "B-SEAT_TYPE", "O", "O", "B-MOVIE_NAME", "O", "B-THEATER_NAME"]`

### Multi-Token Entity Example

**Sentence**: "Show me Dune Part Two"

| Token | BIO Tag |
|-------|---------|
| Show | O |
| me | O |
| Dune | B-MOVIE_NAME |
| Part | I-MOVIE_NAME |
| Two | I-MOVIE_NAME |

**BIO Tags**: `["O", "O", "B-MOVIE_NAME", "I-MOVIE_NAME", "I-MOVIE_NAME"]`

---

## 5. Data Format

### CSV Format

The dataset is stored in CSV format with the following columns:

```
sentence,intent,tokens,BIO_tags
```

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `sentence` | string | Natural language user utterance | "Book 2 vip tickets for Dune at PVR Nexus Mall" |
| `intent` | string | Intent label | "book_ticket" |
| `tokens` | JSON string | List of tokens (words) | `["Book", "2", "vip", "tickets", "for", "Dune", "at", "PVR"]` |
| `BIO_tags` | JSON string | List of BIO tags | `["O", "B-NUM_TICKETS", "B-SEAT_TYPE", "O", "O", "B-MOVIE_NAME", "O", "B-THEATER_NAME"]` |

### JSON Format (Alternative)

```json
{
  "sentence": "Book 2 vip tickets for Dune at PVR",
  "intent": "book_ticket",
  "tokens": ["Book", "2", "vip", "tickets", "for", "Dune", "at", "PVR"],
  "BIO_tags": ["O", "B-NUM_TICKETS", "B-SEAT_TYPE", "O", "O", "B-MOVIE_NAME", "O", "B-THEATER_NAME"]
}
```

---

## 6. Data Generation Process

The dataset is **synthetically generated** using template-based generation with the following approach:

### 6.1 Intent Templates

Each intent has multiple sentence templates with entity placeholders:

```python
INTENT_TEMPLATES = {
    "book_ticket": [
        ["book", (NUM_TICKETS,), (SEAT_TYPE,), "tickets for", (MOVIE_NAME,), "at", (THEATER_NAME,)],
        ["i need", (NUM_TICKETS,), "tickets for", (MOVIE_NAME,), "this", (DATE,)],
        # ... more templates
    ],
    # ... other intents
}
```

### 6.2 Entity Bank

Predefined values for each entity type:

```python
ENTITY_BANK = {
    "MOVIE_NAME": ["Dune Part Two", "Oppenheimer", "Barbie", "Interstellar", ...],
    "CITY": ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", ...],
    "THEATER_NAME": ["PVR Nexus Mall", "INOX City Center", "Cinepolis Central", ...],
    # ... other entities
}
```

### 6.3 Data Augmentation

To increase diversity and robustness:

1. **Noise Application** (45% of examples):
   - Character-level typos: "movie" → "movi"
   - Character deletion: "theater" → "thetre"
   - Character substitution: random character replacement
   - Prefix/suffix insertion: add filler words like "pls", "hey", "um"

2. **Balanced Distribution**:
   - ~125 examples per intent (1000 ÷ 8)
   - Equal representation across intents

3. **Train-Val-Test Split**:
   - Training: 70% (700 examples)
   - Validation: 15% (150 examples)
   - Test: 15% (150 examples)
   - Stratified by intent

---

## 7. Annotation Quality Metrics

### 7.1 Dataset Statistics

- **Total Examples**: 1,000
- **Average Sequence Length**: ~8-10 tokens
- **Maximum Sequence Length**: ~20 tokens
- **Intent Distribution**: Balanced (125 per intent)
- **Entity Density**: ~2-3 entities per sentence on average
- **Token Count**: ~1,200 total entity-labeled tokens

### 7.2 Entity Coverage

| Entity Type | Frequency | % of Dataset |
|------------|-----------|-------------|
| MOVIE_NAME | 850 | 85% |
| THEATER_NAME | 720 | 72% |
| NUM_TICKETS | 680 | 68% |
| CITY | 520 | 52% |
| DATE | 480 | 48% |
| SEAT_TYPE | 450 | 45% |
| TIME | 380 | 38% |
| LANGUAGE | 200 | 20% |
| SEAT_NUMBER | 180 | 18% |
| SCREEN_TYPE | 150 | 15% |

---

## 8. Annotation Guidelines for Manual Verification

When manually annotating or verifying examples, follow these guidelines:

### 8.1 Intent Classification Rules

1. **Always choose the PRIMARY user intent**:
   - If user asks "Show me Dune and book 2 tickets", PRIMARY intent is `book_ticket` (action to perform)
   - Secondary information is captured through entities

2. **Greetings and Farewells**:
   - If the message contains ONLY greeting/goodbye, mark as such
   - If greeting + request, mark the request intent

3. **Ambiguous Cases**:
   - `check_showtime` vs `search_movie`: If asking for specific showtimes, use `check_showtime`; if asking which movies are available, use `search_movie`

### 8.2 Entity Tagging Rules

1. **Tokenization**: Use basic whitespace/punctuation tokenization
   - "PVR Nexus Mall" → ["PVR", "Nexus", "Mall"]
   - Each token gets ONE tag

2. **Multi-word Entities**:
   - First token: B-LABEL
   - Subsequent tokens: I-LABEL
   - "Dune Part Two" → B-MOVIE_NAME, I-MOVIE_NAME, I-MOVIE_NAME

3. **No Overlapping Entities**:
   - Each token belongs to at most one entity
   - If ambiguous, choose the most specific entity

4. **Implicit Entities Are NOT Tagged**:
   - "Book me seats" - `seats` is implied to be seats but not a specific SEAT_NUMBER, so tag as O
   - "In 3D" - if format is implied but not explicitly about movie screen, may be O

5. **Numbers**:
   - "2 tickets" → "2" is NUM_TICKETS
   - "A10" → SEAT_NUMBER (alphanumeric seat code)
   - "May 25" → DATE tokens

---

## 9. Special Cases and Edge Cases

### 9.1 Handling Typos (in Noisy Data)

Original: "Book 2 vip tickets for Dune"  
Noisy: "buk 2 vip tix for dune"

**Tagging remains consistent:**
- "buk" → O (still verb)
- "tix" → O (abbreviated "tickets")
- "dune" → B-MOVIE_NAME (still entity)

### 9.2 Abbreviations

| Abbreviation | Expanded | Entity Type |
|------------|----------|------------|
| PVR | PVR Cinemas | THEATER_NAME |
| AM/PM | Morning/Afternoon/Evening | TIME |
| 2D/3D | Screen format | SCREEN_TYPE |

### 9.3 Colloquial Expressions

| Expression | Intent |
|-----------|--------|
| "Cancel my booking" | `cancel_ticket` |
| "Show showtimes" | `check_showtime` |
| "Pick me seats" | `select_seat` |
| "What's my booking status?" | `check_booking_status` |

---

## 10. Inter-Annotator Agreement

For datasets with multiple annotators, use these metrics:

- **Cohen's Kappa** (for intent labels): Should be > 0.80 for acceptable agreement
- **F1-Score** (for entity tagging): Should be > 0.90 for strict match
- **Entity-level agreement**: Measure on exact span match

---

## 11. Known Limitations

1. **Synthetic Data**: Dataset is artificially generated and may not cover all real-world variations
2. **No Context**: Each example is standalone; no conversation history provided
3. **Entity Bank Constraints**: Only predefined entity values are used (limited generalization)
4. **Balanced Distribution**: Real-world intent distribution would likely be different
5. **Simple Templates**: Real conversations are more varied and complex

---

## 12. Extensions and Future Work

To extend this schema:

1. **Add Dialogue Acts**: Beyond intents, add finer-grained dialogue acts
2. **Add Confidence Scores**: Confidence level for each annotation
3. **Add Dialogue Context**: Multi-turn conversation history
4. **Add Negation/Scope**: Explicit negation markers
5. **Add Coreference**: Links to previously mentioned entities
6. **Domain Expansion**: Other movie booking attributes (ratings, duration, etc.)

---

## 13. Usage in Modeling

### 13.1 For Intent Classification
- Use `intent` column as target label
- Use `tokens` or `sentence` as input features

### 13.2 For Named Entity Recognition (NER)
- Use `BIO_tags` as target labels
- Use `tokens` as input features (tokenized)
- Use BIO tagging scheme for sequence labeling

### 13.3 For Joint Intent + Entity Extraction
- Use both `intent` and `BIO_tags` as multi-task targets
- Single model head for intent classification (sentence-level)
- Second model head for token-level entity classification

### 13.4 Preprocessing Steps
```python
# Load dataset
df = pd.read_csv('dataset.csv')

# Parse JSON columns
df['tokens'] = df['tokens'].apply(json.loads)
df['BIO_tags'] = df['BIO_tags'].apply(json.loads)

# Verify consistency
assert all(len(tokens) == len(tags) for tokens, tags in 
           zip(df['tokens'], df['BIO_tags']))
```

---

## 14. Contact & References

**Dataset Created**: Group 151, June 2026  
**Framework**: PyTorch with Transformers  
**License**: Educational (for course assignment)

For questions about the annotation schema, refer to this document or the notebook's data generation section.

---

## Appendix A: Complete Entity Bank Values

### MOVIE_NAME
- Dune Part Two
- Oppenheimer
- Barbie
- Interstellar
- Jawan
- Leo
- Kalki 2898 AD
- Tiger 3
- Spider Man No Way Home
- The Batman
- Avatar The Way of Water
- Top Gun Maverick

### THEATER_NAME
- PVR Nexus Mall
- INOX City Center
- Cinepolis Central
- Luxe IMAX
- Miraj Cinemas
- Carnival City Mall
- Wave Cinemas
- Raj Mandir

### CITY
- Mumbai
- Delhi
- Bengaluru
- Hyderabad
- Chennai
- Pune
- Kolkata
- Ahmedabad
- Jaipur
- Kochi

### DATE
- today
- tomorrow
- this evening
- tonight
- this weekend
- next Friday
- next Saturday
- 25 May
- 26 May
- 27 May

### TIME
- 10 am
- 11:30 am
- 1 pm
- 3:15 pm
- 5 pm
- 6:45 pm
- 8 pm
- 9:30 pm
- 11 pm

### NUM_TICKETS
- 1, 2, 3, 4, 5, 6

### SEAT_TYPE
- regular
- vip
- premium
- recliner
- balcony

### SEAT_NUMBER
- A10, A11, B7, B12, C3, C4, D8, E15

### LANGUAGE
- English
- Hindi
- Tamil
- Telugu
- Malayalam
- Kannada
- French

### SCREEN_TYPE
- 2D
- 3D
- IMAX
- Dolby
- 4DX

---

**End of Annotation Schema Documentation**

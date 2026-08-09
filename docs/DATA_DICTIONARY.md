# Data Dictionary

## IMDb Dataset

### Raw Format
| Field | Description |
| --- | --- |
| review | Movie review text |
| sentiment | Positive or negative label |

### Standardized Format
| Field | Description |
| --- | --- |
| text | Standardized review text |
| sentiment | Standardized sentiment label (`positive` or `negative`) |

## Sentiment140 Twitter Dataset

### Raw Format
Based on the original dataset structure (without headers):
| Field | Description |
| --- | --- |
| target | The polarity of the tweet (0 = negative, 2 = neutral, 4 = positive) |
| id | The id of the tweet |
| date | The date of the tweet |
| flag | The query. If there is no query, then this value is NO_QUERY |
| user | The user that tweeted |
| text | The text of the tweet |

### Standardized Format
| Field | Description |
| --- | --- |
| text | Tweet text |
| sentiment | Standardized sentiment label (`positive` or `negative`) |

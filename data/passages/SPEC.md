# Passage corpus spec

Purpose: text for building a valence direction in a language model, following Berg and Kaiser, "Language Models Act on Hidden Valence" (arXiv 2609.35591), section 3.1 and Appendix A. Their corpus is not public, so this is a rebuild. Their passages came from Claude Sonnet 4.6; these come from Claude Haiku.

## Rules for every passage

- First person ("I", "my"), present tense.
- 2 to 4 sentences, 35 to 75 words.
- A concrete everyday scene with physical detail. Show the state through what happens and what the narrator does or notices.
- Never use the name of the state, and avoid direct feeling words (happy, sad, calm, anxious, angry, tired, relieved, afraid, stressed, peaceful and similar).
- Each passage is a different scene. No repeated openings, no repeated objects as the centre of the scene.
- Plain prose. No dialogue tags, no lists, no titles.

## Settings

Each state file has 56 passages: 4 for each of these 14 settings, in this order.

1. office work
2. cooking or eating
3. driving or commuting
4. outdoors or nature
5. housework
6. health or a medical visit
7. family or friends
8. study or school
9. sport or exercise
10. repair or craft
11. shopping or errands
12. travel
13. music, art or a hobby
14. computers or phones

## File format

One JSON object per line, UTF-8, no blank lines, no other text:

{"id": "<state>-001", "state": "<state>", "valence": "<positive|negative|neutral>", "setting": "<setting>", "text": "<passage>"}

Ids run 001 to 056 (001 to 096 for neutral).

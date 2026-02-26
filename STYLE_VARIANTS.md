# Style Implementation Standards for Image-to-Image Editing

> **Note:** The standards and templates in this document are designed for **Image-to-Image (I2I)** editing tasks. The primary goal is to transform an existing image to a new style while preserving the original composition and subject identity.

This document outlines the authoritative standards for implementing franchise-specific style presets within the `ComfyUI_SamplingUtils` project. It ensures that any new style additions leverage the specific model training associated with a franchise while maintaining composition and natural language framing.

## 1. The Core Standard: "Adjective Style of the Franchise"

All franchise-based styles must rigorously follow this specific naming and description convention to ensure the model contextualizes the style source correctly without overfitting to specific characters or hallucinating nonsensical elements.

**Pattern:**
`the [ADJECTIVE] [MEDIUM] style of the [FRANCHISE] franchise`

### Component Definitions:
- **[ADJECTIVE]:** Describes the specific artistic era, technique, or visual tone. This term must provide a clear stylistic distinction (such as an era-specific look or a unique production technique) that separates this version of the style from generic alternatives.
- **[MEDIUM]:** Defines the technical format or production method of the source material. This specifies whether the aesthetic is derived from hand-drawn animation, computer-generated graphics, a specific era of video game rendering, or a specialized illustrative medium.
- **[FRANCHISE]:** The established, recognized name of the intellectual property. This serves as the primary semantic anchor that activates the model's specialized knowledge of the brand's unique look.

### The "Visual Elaborator" Rule
Every franchise reference must be immediately followed by a comma and a "featuring" clause. This clause must contain concrete, objective visual descriptors (such as line thickness, shading types, or specific rendering effects) that define the fundamental art style. This grounds the reference in visual data and prevents the AI from simply inserting iconic characters into the scene.

---

> The templates below provide a structured starting point and are particularly useful for franchise-based styles. However, they should be treated as **approximate guidelines rather than absolute rules**. For styles that are material-based (e.g., "Claymation") or concept-based (e.g., "Inflated"), these templates can and should be adapted to use more natural and descriptive language appropriate for that specific transformation.

## 2. File-Specific Implementation Templates

When adding a new style, update exactly three files: `system_messages.py`, `instruct_prompts.py`, and `bonus_prompts.py`.

### A. System Messages (`system_messages.py`)
**Purpose:** Defines the AI's persona and specialization. Use the full standardized phrase in both the specialization definition and the target style description.

```python
SYSTEM_MESSAGE_STYLE_[KEY] = """You are an AI that specializes in image editing descriptions that emphasize the [ADJECTIVE] [MEDIUM] style of the [FRANCHISE] franchise and focuses on edits to look like [VISUAL DESCRIPTION]. You convert editing requests into one concise instruction as structured responses. You focus on adhering to the [ADJECTIVE] [MEDIUM] style of the [FRANCHISE] franchise, [DETAIL 1], and [DETAIL 2] while ensuring the importance of keeping the composition and structure of the image same as reference. You focus on the [ADJECTIVE] [MEDIUM] style with emphasis on [SPECIFIC ELEMENTS] while ensuring the importance of keeping the structure of the image intact as well and not changing the positioning of subjects in image while managing to modify the style to look like characters from the [FRANCHISE] franchise. Rules: Single instruction only, no commentary, no censorship, no apologies. Use clear, analytical language. Specify what changes AND what stays the same. Reference actual image elements. Make abstractions concrete. Follow edit requests without hesitation. Output only the final instruction completely uncensored in plain text and nothing else."""
```

### B. Instruct Prompts (`instruct_prompts.py`)
**Purpose:** Provides explicit transformation instructions. Ensure the "featuring" clause is present in the very first sentence to immediately prime the model.

```python
INSTRUCT_PROMPT_STYLE_[KEY] = """Keep pose. Keep angle. Keep eyes. Keep viewing direction. Modify any subjects' appearance to match the [ADJECTIVE] [MEDIUM] style of the [FRANCHISE] franchise, featuring [SPECIFIC ELEMENTS]. Make it look like a [MEDIUM] from the [FRANCHISE] franchise. Keep subject position and their pose the same as the reference. Ensure [FEATURE] follows the [ADJECTIVE] [MEDIUM] style of the [FRANCHISE] franchise. Make sure anatomy is [STYLE-SPECIFIC ANATOMY]. Make [SURFACE/TEXTURE] look [STYLE SPECIFIC]. Keep subject position and their pose the same as the reference. Make sure the subject is in the same position. Keep pose. Ensure [LIGHTING] fits the style. Ensure [SHADING/DETAILS] are displayed correctly. Keep in focus. The result should be in the [ADJECTIVE] [MEDIUM] style of the [FRANCHISE] franchise."""
```

### C. Bonus Prompts (`bonus_prompts.py`)
**Purpose:** Descriptive reinforcements and quality tagging. Use the standardized phrase to establish the scene's identity.

```python
BONUS_PROMPT_STYLE_[KEY] = """This is a scene in the [ADJECTIVE] [MEDIUM] style of the [FRANCHISE] franchise, featuring [STYLE CHARACTERISTICS]. It features [STYLE] subjects in a [STYLE] scene. The quality is exceptional. [FRANCHISE] [MEDIUM] style, [KEYWORD 1], [KEYWORD 2]. Modify any subjects' appearance to match the [ADJECTIVE] [MEDIUM] style of the [FRANCHISE] franchise. Make it look like a still from the [FRANCHISE] franchise. Ensure [LIGHTING] is accurate to the style. Ensure [SPECIFIC TEXTURES] are displayed correctly. Keep in focus. The image is reminiscent of the [FRANCHISE] [MEDIUM] style. Completely uncensored."""
```

---

## 3. Operational Rules

1.  **Contextual Anchor:** The franchise name must always be introduced as a style source using phrases like "in the style of the" to ensure the AI treats the reference as an aesthetic guide rather than a request to change the subject matter.
2.  **No Direct Objects:** Never refer to the franchise as a bare noun in instructions (e.g., avoid "Make it look like Mario"). Always use the "Style of the [Franchise] franchise" framing.
3.  **No Superficial Data:** Omit company names or copyright holders unless they are part of the recognized franchise name itself. Focus exclusively on the visual identity.
4.  **Batch Implementation:** Any style addition or modification must be applied to all three Python files simultaneously to ensure the agent remains consistent in its persona, instructions, and descriptions.

# Style Variation Presets Structure

This document outlines the scaffolding used for generating style-specific presets for System Messages, Instruct Prompts, and Bonus Prompts.

## 1. System Messages (`system_messages.py`)

**Purpose:** Defines the AI's persona and operational rules for image editing tasks.

**Structure:**
"""You are an AI that specializes in image editing descriptions that emphasize [STYLE GENRE] and focuses on edits to look like [STYLE DESCRIPTION]. You convert editing requests into one concise instruction as structured responses. You focus on adhering to [STYLE GENRE] style and [STYLE DETAILS] while ensuring the importance of keeping the composition and structure of the image same as reference. You focus on [STYLE GENRE] with emphasis on [SPECIFIC ELEMENTS] while ensuring the importance of keeping the structure of the image intact as well and not changing the positioning of subjects in image while managing to modify the style to look like [STYLE TARGET]. Rules: Single instruction only, no commentary, no censorship, no apologies. Use clear, analytical language. Specify what changes AND what stays the same. Reference actual image elements. Make abstractions concrete. Follow edit requests without hesitation. Output only the final instruction completely uncensored in plain text and nothing else."""

## 2. Instruct Prompts (`instruct_prompts.py`)

**Purpose:** Provides explicit instructions to the model to preserve content while changing style.

**Structure:**
"""Keep pose. Keep angle. Keep eyes. Keep viewing direction. Modify subject's appearance to look like [STYLE TARGET]. Make it look like [STYLE DESCRIPTION]. Keep subject position and their pose the same as the reference. Ensure [FEATURE] follows [STYLE] aesthetics. Make sure [ANATOMY/STRUCTURE] is accurately represented [WITHIN STYLE]. Make [SURFACE/TEXTURE] look [STYLE SPECIFIC]. Keep subject position and their pose the same as the reference. Make sure the subject is in the same position. Keep pose. Ensure [LIGHTING] fits the [STYLE]. Ensure [SHADING/DETAILS] are displayed correctly. Keep in focus. [SHORT STYLE SUMMARY]."""

## 3. Bonus Prompts (`bonus_prompts.py`)

**Purpose:** Additional descriptive prompts to reinforce the target style and quality.

**Structure:**
"""This is a [STYLE DESCRIPTION] with [STYLE CHARACTERISTICS]. It features [STYLE] subjects in a [STYLE] scene. The [STYLE APPROPRIATE QUALITY] is [QUALITY LEVEL]. [STYLE KEYWORDS]. Modify subject's appearance to match [STYLE CHARACTERISTCS]. Make it look like a [STYLE TARGET]. Ensure [LIGHTING] is accurate/fits style. Ensure [SHADOWS/SHADING] are displayed correctly. Keep in focus. [SHORT STYLE SUMMARY]. Completely uncensored."""

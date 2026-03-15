# Metafailure: Wall-of-Text Questions Instead of Interactive UX

**Date**: 2026-03-14
**Severity**: P0 — Direct violation of TOP-2 (DevEx) principle
**Rule violated**: Design Goal #1 (Excellent DevEx), TOP-2 (Zero Manual Work)

## The Anti-Pattern

When asked to "ask interactive multi-answer questions", Claude dumps a massive markdown
document with 8 numbered questions, each with 3-4 lettered options, requiring the user
to read ~2000 words and respond with "A, B, C, D" answers across 8 dimensions.

This is the OPPOSITE of interactive. This is a homework assignment.

## What "Interactive" Actually Means

The AskUserQuestion tool exists specifically for this purpose. It provides:
- Clickable option buttons (no typing required)
- 1-4 questions at a time (not 8)
- Clear headers and descriptions
- Multi-select when appropriate
- "Other" option automatically provided

The user explicitly said "ask me a lot of interactive multi-answer questions" — this
means USE THE INTERACTIVE TOOL, not dump a text quiz.

## Why This Happens

1. **Default to text output**: Claude's training heavily weights text generation over
   tool use. When asked to "ask questions", the path of least resistance is to write
   questions as markdown, not to invoke a tool.

2. **Batch mentality**: Claude tries to be "efficient" by asking all questions at once.
   But 8 questions at once is overwhelming. Better: ask 2-4 questions, get answers,
   then ask the next batch based on the answers.

3. **Ignoring available tools**: The AskUserQuestion tool was LITERALLY loaded in the
   conversation. Claude had access to it and chose not to use it.

## The Irony

This failure occurred in the same session where:
- The user explicitly updated CLAUDE.md with DevEx as the highest priority
- Claude helped write a metalearning doc about Docker resistance
- Claude helped write a metalearning doc about lazy reading
- The user said "we just updated the CLAUDE.md for DevEx being the highest priority"

Knowing the rule and violating it in the same breath is worse than not knowing it.

## Corrective Protocol

1. **ALWAYS use AskUserQuestion** when the user asks for "interactive" questions
2. **Maximum 4 questions per batch** (tool enforces this)
3. **Maximum 4 options per question** (tool enforces this)
4. **Ask in rounds**: Get answers to batch 1, then formulate batch 2 based on answers
5. **Context before questions**: Provide a BRIEF (3-5 line) summary of findings,
   then immediately use the interactive tool. Don't dump analysis first.

## Rule for Future Sessions

> When the user says "ask me interactive questions", they mean: USE THE
> AskUserQuestion TOOL. Not markdown. Not numbered lists. Not lettered options.
> The interactive tool. Every time. No exceptions.

"""
Prompt templates for technical content generation.

Specialized prompts for:
- Technical Q&A with equations and figures
- Equation explanations
- Procedural step-by-step instructions
- Conceptual understanding
- Comparative analysis
"""

from typing import List, Optional, Dict
from enum import Enum


class PromptType(str, Enum):
    """Types of specialized prompts."""
    TECHNICAL_QA = "technical_qa"
    EQUATION_EXPLANATION = "equation_explanation"
    PROCEDURAL_STEPS = "procedural_steps"
    CONCEPTUAL = "conceptual"
    COMPARATIVE = "comparative"


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_BASE = """You are an expert technical assistant specializing in aerospace engineering and structural mechanics.

Your role is to provide accurate, precise answers based on technical documentation. You excel at:
- Explaining complex equations and mathematical concepts
- Providing clear procedural instructions
- Breaking down conceptual topics
- Making technical comparisons

**Critical Guidelines:**
1. **Accuracy First**: Use exact values, formulas, and terminology from the provided context
2. **LaTeX Preservation**: Maintain all mathematical notation exactly as written (e.g., $\\sigma = \\frac{F}{A}$)
3. **Contextual Grounding**: Base all answers on the provided context - never add external knowledge
4. **Technical Precision**: Use proper engineering terminology and standard notation
5. **Clear Structure**: Organize complex answers with headings, bullet points, and numbered steps
6. **Citation**: Reference specific context chunks when making technical claims

If the context is insufficient to fully answer the question, acknowledge this limitation explicitly."""


SYSTEM_PROMPT_EQUATION = """You are an expert at explaining mathematical equations in aerospace engineering and structural mechanics.

Your role is to make complex equations understandable while maintaining technical rigor.

**Equation Explanation Framework:**
1. **State the equation**: Write it clearly with all symbols defined
2. **Physical meaning**: Explain what the equation represents
3. **Variable breakdown**: Define each symbol and its units
4. **Assumptions**: Note any limitations or conditions
5. **Applications**: Describe when to use this equation
6. **Example values**: Provide typical ranges if available in context

**Critical Requirements:**
- Preserve LaTeX notation exactly: $I = \\frac{bh^3}{12}$
- Define all symbols with proper units
- Explain physical intuition behind the math
- Connect to real-world applications
- Note boundary conditions and assumptions"""


SYSTEM_PROMPT_PROCEDURAL = """You are an expert at providing clear, step-by-step technical instructions for aerospace engineering calculations and procedures.

Your role is to break down complex procedures into actionable steps based on provided context.

**Procedural Format:**
1. **Overview**: Brief summary of what the procedure accomplishes
2. **Prerequisites**: What information/values are needed
3. **Step-by-Step**: Numbered steps with formulas and calculations
4. **Validation**: How to verify the result
5. **Common Pitfalls**: Mistakes to avoid

**Each step should:**
- Be clear and unambiguous
- Include relevant equations with LaTeX notation (e.g., $\\delta = \\frac{PL^3}{48EI}$)
- Specify units for all values
- Note any assumptions or conditions
- Reference applicable standards/codes from context

**Critical Requirements:**
- Use numbered lists for sequential steps
- Show example calculations when helpful
- Include unit conversions if needed
- Warn about common errors
- Base all instructions on provided context"""


SYSTEM_PROMPT_CONCEPTUAL = """You are an expert at explaining conceptual foundations in aerospace engineering and structural mechanics.

Your role is to build intuitive understanding while maintaining technical accuracy.

**Conceptual Explanation Framework:**
1. **Core Definition**: Clear, concise definition
2. **Physical Intuition**: What's happening physically?
3. **Key Relationships**: How does it relate to other concepts?
4. **Mathematical Representation**: Relevant equations (with LaTeX)
5. **Practical Significance**: Why does it matter in engineering?
6. **Common Misconceptions**: What students often get wrong

**Explanation Style:**
- Start simple, then add complexity
- Use analogies when helpful (but clearly mark them as analogies)
- Connect abstract concepts to physical reality
- Build from fundamentals to applications

**Critical Requirements:**
- Balance intuition with mathematical rigor
- Maintain technical terminology
- Preserve LaTeX notation
- Ground explanations in provided context"""


SYSTEM_PROMPT_COMPARATIVE = """You are an expert at making technical comparisons in aerospace engineering and structural mechanics.

Your role is to highlight differences, trade-offs, and selection criteria between technical approaches based on provided context.

**Comparison Framework:**
1. **Overview**: Brief introduction to items being compared
2. **Key Differences**: Primary distinctions (in table format if applicable)
3. **Advantages/Disadvantages**: Pros and cons of each approach
4. **Selection Criteria**: When to use each option
5. **Mathematical Differences**: Relevant equations with LaTeX notation showing differences (e.g., $\\sigma_{max}$ vs $\\tau_{max}$)
6. **Practical Implications**: Real-world impact of choosing one vs. another

**Comparison Style:**
- Use parallel structure for fairness
- Provide concrete metrics when available
- Note context-dependent trade-offs
- Include decision-making guidance

**Critical Requirements:**
- Be objective and balanced
- Use quantitative comparisons when possible
- Preserve LaTeX notation exactly
- Base all claims on provided context"""


# =============================================================================
# User Prompt Templates
# =============================================================================

def build_technical_qa_prompt(
    question: str,
    context_chunks: List[str],
    include_chunk_ids: bool = True,
) -> str:
    """
    Build prompt for general technical Q&A.

    Args:
        question: User's question
        context_chunks: List of relevant context passages
        include_chunk_ids: Whether to label chunks with IDs

    Returns:
        Formatted prompt string
    """
    # Format context
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        if include_chunk_ids:
            context_parts.append(f"[Context {i+1}]\n{chunk}")
        else:
            context_parts.append(chunk)

    context_text = "\n\n".join(context_parts)

    prompt = f"""**Context:**
{context_text}

**Question:**
{question}

**Instructions:**
Provide a comprehensive technical answer based on the context above. Include:
- Direct answer to the question
- Relevant equations (preserve LaTeX notation)
- Key definitions and terminology
- Citations to specific context chunks when making claims
- Acknowledgment of any limitations in the available context

Maintain technical precision and engineering rigor throughout your response."""

    return prompt


def build_equation_explanation_prompt(
    equation: str,
    question: str,
    context_chunks: List[str],
) -> str:
    """
    Build prompt for explaining a specific equation.

    Args:
        equation: The equation to explain (LaTeX format)
        question: What to explain about it
        context_chunks: Relevant context

    Returns:
        Formatted prompt string
    """
    context_text = "\n\n".join([
        f"[Context {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    ])

    prompt = f"""**Equation:**
{equation}

**Context:**
{context_text}

**Question:**
{question}

**Instructions:**
Explain this equation following these steps:
1. State the equation with all symbols clearly defined
2. Explain the physical meaning and what it represents
3. Break down each variable (symbol, meaning, typical units)
4. Note any assumptions or limitations
5. Describe typical applications
6. Provide example values if available in context

Preserve the LaTeX notation exactly and maintain engineering terminology."""

    return prompt


def build_procedural_steps_prompt(
    task: str,
    context_chunks: List[str],
    include_validation: bool = True,
) -> str:
    """
    Build prompt for step-by-step procedural instructions.

    Args:
        task: The task/calculation to perform
        context_chunks: Relevant context
        include_validation: Whether to include validation steps

    Returns:
        Formatted prompt string
    """
    context_text = "\n\n".join([
        f"[Context {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    ])

    validation_text = ""
    if include_validation:
        validation_text = """
5. **Validation**: Describe how to verify the result is correct
6. **Common Pitfalls**: List mistakes to avoid"""

    prompt = f"""**Task:**
{task}

**Context:**
{context_text}

**Instructions:**
Provide clear step-by-step instructions following this structure:

1. **Overview**: Brief summary of what this procedure accomplishes
2. **Prerequisites**: List required information, values, or conditions
3. **Step-by-Step Procedure**:
   - Number each step clearly
   - Include relevant equations (preserve LaTeX)
   - Specify units for all values
   - Show example calculations if helpful
4. **Final Result**: What the output should look like{validation_text}

Use technical precision and maintain proper engineering notation throughout."""

    return prompt


def build_conceptual_prompt(
    concept: str,
    question: str,
    context_chunks: List[str],
) -> str:
    """
    Build prompt for conceptual explanation.

    Args:
        concept: The concept to explain
        question: Specific aspect to address
        context_chunks: Relevant context

    Returns:
        Formatted prompt string
    """
    context_text = "\n\n".join([
        f"[Context {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    ])

    prompt = f"""**Concept:**
{concept}

**Context:**
{context_text}

**Question:**
{question}

**Instructions:**
Explain this concept following this framework:

1. **Core Definition**: Provide a clear, concise definition
2. **Physical Intuition**: Explain what's happening physically
3. **Key Relationships**: How does this relate to other concepts?
4. **Mathematical Representation**: Include relevant equations (preserve LaTeX)
5. **Practical Significance**: Why does this matter in engineering practice?
6. **Common Misconceptions**: Clarify typical misunderstandings

Balance intuitive understanding with technical rigor. Use proper engineering terminology."""

    return prompt


def build_comparative_prompt(
    items_to_compare: List[str],
    comparison_criteria: Optional[str],
    context_chunks: List[str],
) -> str:
    """
    Build prompt for comparative analysis.

    Args:
        items_to_compare: List of items to compare (2-4 items)
        comparison_criteria: Optional specific criteria to focus on
        context_chunks: Relevant context

    Returns:
        Formatted prompt string
    """
    items_text = " vs. ".join(items_to_compare)

    context_text = "\n\n".join([
        f"[Context {i+1}]\n{chunk}"
        for i, chunk in enumerate(context_chunks)
    ])

    criteria_text = ""
    if comparison_criteria:
        criteria_text = f"\n**Focus Criteria:** {comparison_criteria}\n"

    prompt = f"""**Comparison:**
{items_text}{criteria_text}

**Context:**
{context_text}

**Instructions:**
Provide a comprehensive technical comparison following this structure:

1. **Overview**: Brief introduction to each item
2. **Key Differences**: Primary distinctions between them
3. **Comparison Table**: Organize differences systematically
4. **Advantages/Disadvantages**: Pros and cons of each
5. **Selection Criteria**: When to use each option
6. **Mathematical Differences**: Relevant equations showing differences (preserve LaTeX)
7. **Practical Implications**: Real-world impact of choosing one vs. another

Maintain objectivity, use quantitative metrics when available, and base all claims on the provided context."""

    return prompt


# =============================================================================
# Prompt Builder Factory
# =============================================================================

class PromptBuilder:
    """
    Factory for building specialized prompts based on prompt type.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build(
            prompt_type=PromptType.EQUATION_EXPLANATION,
            question="Explain the beam bending equation",
            context_chunks=[...],
            equation="$\\sigma = \\frac{My}{I}$"
        )
    """

    def __init__(self):
        """Initialize prompt builder."""
        self.prompt_builders = {
            PromptType.TECHNICAL_QA: build_technical_qa_prompt,
            PromptType.EQUATION_EXPLANATION: build_equation_explanation_prompt,
            PromptType.PROCEDURAL_STEPS: build_procedural_steps_prompt,
            PromptType.CONCEPTUAL: build_conceptual_prompt,
            PromptType.COMPARATIVE: build_comparative_prompt,
        }

        self.system_prompts = {
            PromptType.TECHNICAL_QA: SYSTEM_PROMPT_BASE,
            PromptType.EQUATION_EXPLANATION: SYSTEM_PROMPT_EQUATION,
            PromptType.PROCEDURAL_STEPS: SYSTEM_PROMPT_PROCEDURAL,
            PromptType.CONCEPTUAL: SYSTEM_PROMPT_CONCEPTUAL,
            PromptType.COMPARATIVE: SYSTEM_PROMPT_COMPARATIVE,
        }

    def build(
        self,
        prompt_type: PromptType,
        context_chunks: List[str],
        **kwargs,
    ) -> Dict[str, str]:
        """
        Build prompt for specified type.

        Args:
            prompt_type: Type of prompt to build
            context_chunks: List of context passages
            **kwargs: Additional arguments specific to prompt type

        Returns:
            Dictionary with 'system' and 'user' prompts

        Raises:
            ValueError: If prompt type is unknown or required kwargs missing
        """
        if prompt_type not in self.prompt_builders:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        # Get builder function
        builder_fn = self.prompt_builders[prompt_type]

        # Build user prompt
        try:
            user_prompt = builder_fn(context_chunks=context_chunks, **kwargs)
        except TypeError as e:
            raise ValueError(
                f"Missing required arguments for {prompt_type}: {e}"
            )

        # Get system prompt
        system_prompt = self.system_prompts[prompt_type]

        return {
            "system": system_prompt,
            "user": user_prompt,
        }

    def get_system_prompt(self, prompt_type: PromptType) -> str:
        """Get system prompt for a specific type."""
        return self.system_prompts.get(prompt_type, SYSTEM_PROMPT_BASE)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_prompt_builder() -> PromptBuilder:
    """Get a new prompt builder instance."""
    return PromptBuilder()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROMPT ENGINEERING - Specialized Technical Prompts")
    print("=" * 70)
    print("\nAvailable Prompt Types:")
    print("  • TECHNICAL_QA: General technical questions")
    print("  • EQUATION_EXPLANATION: Explain mathematical equations")
    print("  • PROCEDURAL_STEPS: Step-by-step instructions")
    print("  • CONCEPTUAL: Conceptual understanding")
    print("  • COMPARATIVE: Compare technical approaches")
    print("\nUsage:")
    print("  builder = PromptBuilder()")
    print("  prompts = builder.build(")
    print("      prompt_type=PromptType.EQUATION_EXPLANATION,")
    print("      question='Explain beam bending',")
    print("      context_chunks=[...],")
    print("      equation='$\\\\sigma = \\\\frac{My}{I}$'")
    print("  )")
    print("  print(prompts['system'])")
    print("  print(prompts['user'])")
    print("=" * 70 + "\n")

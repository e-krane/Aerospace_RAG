"""
Tests for prompt engineering module.

Tests:
- Technical Q&A prompts
- Equation explanation prompts
- Procedural steps prompts
- Conceptual explanation prompts
- Comparative analysis prompts
- Prompt builder factory
"""

import pytest

from src.llm.prompts import (
    PromptBuilder,
    PromptType,
    build_technical_qa_prompt,
    build_equation_explanation_prompt,
    build_procedural_steps_prompt,
    build_conceptual_prompt,
    build_comparative_prompt,
    get_prompt_builder,
)


class TestTechnicalQAPrompt:
    """Test technical Q&A prompt generation."""

    def test_basic_qa_prompt(self):
        """Test basic Q&A prompt with context."""
        question = "What is beam bending?"
        context = ["Beam bending occurs when...", "The stress distribution..."]

        prompt = build_technical_qa_prompt(question, context)

        assert "What is beam bending?" in prompt
        assert "[Context 1]" in prompt
        assert "[Context 2]" in prompt
        assert "Beam bending occurs when..." in prompt

    def test_qa_prompt_without_chunk_ids(self):
        """Test Q&A prompt without chunk IDs."""
        question = "What is stress?"
        context = ["Stress is force per unit area"]

        prompt = build_technical_qa_prompt(
            question,
            context,
            include_chunk_ids=False
        )

        assert "What is stress?" in prompt
        assert "[Context 1]" not in prompt
        assert "Stress is force per unit area" in prompt

    def test_qa_prompt_instructions(self):
        """Test Q&A prompt includes proper instructions."""
        question = "Test question"
        context = ["Test context"]

        prompt = build_technical_qa_prompt(question, context)

        assert "LaTeX notation" in prompt
        assert "Citations" in prompt or "citations" in prompt
        assert "technical" in prompt.lower()


class TestEquationExplanationPrompt:
    """Test equation explanation prompt generation."""

    def test_equation_prompt_structure(self):
        """Test equation explanation prompt structure."""
        equation = "$\\sigma = \\frac{F}{A}$"
        question = "Explain this stress formula"
        context = ["Stress is defined as..."]

        prompt = build_equation_explanation_prompt(equation, question, context)

        assert equation in prompt
        assert question in prompt
        assert "[Context 1]" in prompt
        assert "Stress is defined as..." in prompt

    def test_equation_prompt_instructions(self):
        """Test equation prompt includes explanation steps."""
        equation = "$I = \\frac{bh^3}{12}$"
        question = "Explain moment of inertia"
        context = ["Moment of inertia..."]

        prompt = build_equation_explanation_prompt(equation, question, context)

        assert "symbols" in prompt.lower()
        assert "physical meaning" in prompt.lower()
        assert "assumptions" in prompt.lower()
        assert "applications" in prompt.lower()

    def test_equation_prompt_latex_preservation(self):
        """Test LaTeX notation is preserved."""
        equation = "$$\\tau = \\frac{VQ}{Ib}$$"
        question = "Explain shear stress"
        context = ["Shear stress formula..."]

        prompt = build_equation_explanation_prompt(equation, question, context)

        assert "$$\\tau = \\frac{VQ}{Ib}$$" in prompt
        assert "LaTeX" in prompt


class TestProceduralStepsPrompt:
    """Test procedural steps prompt generation."""

    def test_procedural_prompt_structure(self):
        """Test procedural prompt structure."""
        task = "Calculate beam deflection"
        context = ["Deflection formula: $\\delta = \\frac{PL^3}{48EI}$"]

        prompt = build_procedural_steps_prompt(task, context)

        assert task in prompt
        assert "[Context 1]" in prompt
        assert "Deflection formula" in prompt

    def test_procedural_prompt_with_validation(self):
        """Test procedural prompt includes validation."""
        task = "Calculate stress"
        context = ["Stress = F/A"]

        prompt = build_procedural_steps_prompt(
            task,
            context,
            include_validation=True
        )

        assert "Validation" in prompt
        assert "Common Pitfalls" in prompt or "mistakes" in prompt.lower()

    def test_procedural_prompt_without_validation(self):
        """Test procedural prompt without validation."""
        task = "Calculate strain"
        context = ["Strain = ΔL/L₀"]

        prompt = build_procedural_steps_prompt(
            task,
            context,
            include_validation=False
        )

        # Should still have main structure but no validation section
        assert "Step-by-Step" in prompt
        assert "Prerequisites" in prompt

    def test_procedural_prompt_instructions(self):
        """Test procedural prompt includes clear instructions."""
        task = "Test task"
        context = ["Test context"]

        prompt = build_procedural_steps_prompt(task, context)

        assert "Number each step" in prompt or "numbered" in prompt.lower()
        assert "units" in prompt.lower()
        assert "equations" in prompt.lower()


class TestConceptualPrompt:
    """Test conceptual explanation prompt generation."""

    def test_conceptual_prompt_structure(self):
        """Test conceptual prompt structure."""
        concept = "Hooke's Law"
        question = "Explain the relationship between stress and strain"
        context = ["Hooke's Law states that stress is proportional to strain"]

        prompt = build_conceptual_prompt(concept, question, context)

        assert concept in prompt
        assert question in prompt
        assert "[Context 1]" in prompt
        assert "Hooke's Law states" in prompt

    def test_conceptual_prompt_framework(self):
        """Test conceptual prompt includes explanation framework."""
        concept = "Moment of Inertia"
        question = "What is moment of inertia?"
        context = ["Moment of inertia measures..."]

        prompt = build_conceptual_prompt(concept, question, context)

        assert "Core Definition" in prompt
        assert "Physical Intuition" in prompt
        assert "Key Relationships" in prompt
        assert "Mathematical Representation" in prompt
        assert "Practical Significance" in prompt
        assert "Common Misconceptions" in prompt


class TestComparativePrompt:
    """Test comparative analysis prompt generation."""

    def test_comparative_prompt_structure(self):
        """Test comparative prompt structure."""
        items = ["Euler-Bernoulli Beam Theory", "Timoshenko Beam Theory"]
        context = ["Euler-Bernoulli assumes...", "Timoshenko accounts for shear..."]

        prompt = build_comparative_prompt(items, None, context)

        assert "Euler-Bernoulli Beam Theory" in prompt
        assert "Timoshenko Beam Theory" in prompt
        assert "[Context 1]" in prompt
        assert "[Context 2]" in prompt

    def test_comparative_prompt_with_criteria(self):
        """Test comparative prompt with specific criteria."""
        items = ["Steel", "Aluminum"]
        criteria = "Compare strength-to-weight ratio"
        context = ["Steel has high strength...", "Aluminum is lightweight..."]

        prompt = build_comparative_prompt(items, criteria, context)

        assert "Steel" in prompt
        assert "Aluminum" in prompt
        assert criteria in prompt

    def test_comparative_prompt_framework(self):
        """Test comparative prompt includes comparison framework."""
        items = ["Method A", "Method B"]
        context = ["Method A is..."]

        prompt = build_comparative_prompt(items, None, context)

        assert "Key Differences" in prompt
        assert "Advantages" in prompt or "Disadvantages" in prompt
        assert "Selection Criteria" in prompt
        assert "Practical Implications" in prompt

    def test_comparative_prompt_multiple_items(self):
        """Test comparison with 3+ items."""
        items = ["Option A", "Option B", "Option C"]
        context = ["A...", "B...", "C..."]

        prompt = build_comparative_prompt(items, None, context)

        # Should format items correctly
        assert "Option A" in prompt
        assert "Option B" in prompt
        assert "Option C" in prompt


class TestPromptBuilder:
    """Test prompt builder factory."""

    def test_builder_initialization(self):
        """Test builder initializes correctly."""
        builder = PromptBuilder()

        assert len(builder.prompt_builders) == 5
        assert len(builder.system_prompts) == 5

    def test_build_technical_qa(self):
        """Test building technical Q&A prompt."""
        builder = PromptBuilder()

        prompts = builder.build(
            prompt_type=PromptType.TECHNICAL_QA,
            question="What is stress?",
            context_chunks=["Stress is F/A"],
        )

        assert "system" in prompts
        assert "user" in prompts
        assert "What is stress?" in prompts["user"]
        assert "aerospace engineering" in prompts["system"].lower()

    def test_build_equation_explanation(self):
        """Test building equation explanation prompt."""
        builder = PromptBuilder()

        prompts = builder.build(
            prompt_type=PromptType.EQUATION_EXPLANATION,
            equation="$\\sigma = F/A$",
            question="Explain stress formula",
            context_chunks=["Stress formula..."],
        )

        assert "system" in prompts
        assert "user" in prompts
        assert "$\\sigma = F/A$" in prompts["user"]

    def test_build_procedural_steps(self):
        """Test building procedural steps prompt."""
        builder = PromptBuilder()

        prompts = builder.build(
            prompt_type=PromptType.PROCEDURAL_STEPS,
            task="Calculate deflection",
            context_chunks=["Deflection = PL³/48EI"],
        )

        assert "system" in prompts
        assert "user" in prompts
        assert "Calculate deflection" in prompts["user"]
        assert "step-by-step" in prompts["system"].lower()

    def test_build_conceptual(self):
        """Test building conceptual explanation prompt."""
        builder = PromptBuilder()

        prompts = builder.build(
            prompt_type=PromptType.CONCEPTUAL,
            concept="Hooke's Law",
            question="Explain the concept",
            context_chunks=["Hooke's Law..."],
        )

        assert "system" in prompts
        assert "user" in prompts
        assert "Hooke's Law" in prompts["user"]

    def test_build_comparative(self):
        """Test building comparative analysis prompt."""
        builder = PromptBuilder()

        prompts = builder.build(
            prompt_type=PromptType.COMPARATIVE,
            items_to_compare=["Option A", "Option B"],
            comparison_criteria=None,
            context_chunks=["A...", "B..."],
        )

        assert "system" in prompts
        assert "user" in prompts
        assert "Option A" in prompts["user"]
        assert "Option B" in prompts["user"]

    def test_build_with_missing_args(self):
        """Test builder fails gracefully with missing args."""
        builder = PromptBuilder()

        with pytest.raises(ValueError):
            # Missing required 'question' argument
            builder.build(
                prompt_type=PromptType.TECHNICAL_QA,
                context_chunks=["Context"],
            )

    def test_get_system_prompt(self):
        """Test getting system prompt for a type."""
        builder = PromptBuilder()

        system_prompt = builder.get_system_prompt(PromptType.EQUATION_EXPLANATION)

        assert "equation" in system_prompt.lower()
        assert "LaTeX" in system_prompt

    def test_get_prompt_builder_convenience(self):
        """Test convenience function."""
        builder = get_prompt_builder()

        assert isinstance(builder, PromptBuilder)


class TestSystemPrompts:
    """Test system prompt content."""

    def test_all_prompts_mention_latex(self):
        """Test all system prompts address LaTeX preservation."""
        builder = PromptBuilder()

        for prompt_type in PromptType:
            system_prompt = builder.get_system_prompt(prompt_type)
            assert "LaTeX" in system_prompt or "latex" in system_prompt.lower()

    def test_all_prompts_mention_context(self):
        """Test all system prompts emphasize using context."""
        builder = PromptBuilder()

        for prompt_type in PromptType:
            system_prompt = builder.get_system_prompt(prompt_type)
            assert "context" in system_prompt.lower()

    def test_all_prompts_mention_technical(self):
        """Test all system prompts emphasize technical accuracy."""
        builder = PromptBuilder()

        for prompt_type in PromptType:
            system_prompt = builder.get_system_prompt(prompt_type)
            assert (
                "technical" in system_prompt.lower() or
                "engineering" in system_prompt.lower() or
                "accurate" in system_prompt.lower() or
                "precision" in system_prompt.lower()
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

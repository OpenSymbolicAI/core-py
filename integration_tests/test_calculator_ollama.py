"""Integration tests for ScientificCalculator with Ollama.

These tests require Ollama to be running locally with the gpt-oss:20b model.
Run with: uv run pytest integration_tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add examples/calculator to path so we can import the calculator
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "calculator"))

from calculator import ScientificCalculator

from opensymbolicai.llm import LLMConfig, Provider

MODEL = "gpt-oss:20b"


@pytest.fixture
def calculator() -> ScientificCalculator:
    """Create a ScientificCalculator with Ollama backend."""
    config = LLMConfig(
        provider=Provider.OLLAMA,
        model=MODEL,
    )
    return ScientificCalculator(llm=config)


def check_ollama_available() -> bool:
    """Check if Ollama is running and the model is available."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"User-Agent": "opensymbolicai/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


pytestmark = pytest.mark.skipif(
    not check_ollama_available(),
    reason="Ollama not available at localhost:11434",
)


class TestBasicArithmetic:
    """Test basic arithmetic operations."""

    def test_addition(self, calculator: ScientificCalculator) -> None:
        """Test simple addition."""
        result = calculator.run("Add 5 and 3")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert result.result == 8

    def test_subtraction(self, calculator: ScientificCalculator) -> None:
        """Test simple subtraction."""
        result = calculator.run("Subtract 10 from 25")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert result.result == 15

    def test_multiplication(self, calculator: ScientificCalculator) -> None:
        """Test simple multiplication."""
        result = calculator.run("Multiply 6 by 7")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert result.result == 42

    def test_division(self, calculator: ScientificCalculator) -> None:
        """Test simple division."""
        result = calculator.run("Divide 100 by 4")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert result.result == 25.0


class TestPowersAndRoots:
    """Test power and root operations."""

    def test_square(self, calculator: ScientificCalculator) -> None:
        """Test squaring a number."""
        result = calculator.run("What is 5 raised to the power of 2?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert result.result == 25

    def test_cube(self, calculator: ScientificCalculator) -> None:
        """Test cubing a number."""
        result = calculator.run("Calculate 3 to the power of 3")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert result.result == 27

    def test_square_root(self, calculator: ScientificCalculator) -> None:
        """Test square root."""
        result = calculator.run("What is the square root of 144?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert result.result == 12


class TestTrigonometry:
    """Test trigonometric functions."""

    def test_sine_90_degrees(self, calculator: ScientificCalculator) -> None:
        """Test sine of 90 degrees."""
        result = calculator.run("What is sine of 90 degrees?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert abs(result.result - 1.0) < 0.0001

    def test_cosine_0_degrees(self, calculator: ScientificCalculator) -> None:
        """Test cosine of 0 degrees."""
        result = calculator.run("What is cosine of 0 degrees?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert abs(result.result - 1.0) < 0.0001

    def test_cosine_45_degrees(self, calculator: ScientificCalculator) -> None:
        """Test cosine of 45 degrees."""
        result = calculator.run("What is cosine of 45 degrees?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert abs(result.result - 0.7071) < 0.001


class TestComplexCalculations:
    """Test multi-step calculations."""

    def test_circle_area(self, calculator: ScientificCalculator) -> None:
        """Test calculating circle area."""
        result = calculator.run("Calculate the area of a circle with radius 5")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        # Area = pi * r^2 = pi * 25 ≈ 78.54
        assert abs(result.result - 78.54) < 0.1

    def test_pythagorean(self, calculator: ScientificCalculator) -> None:
        """Test Pythagorean theorem."""
        result = calculator.run(
            "Calculate the hypotenuse of a right triangle with sides 3 and 4"
        )
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert result.result == 5.0

    def test_percentage(self, calculator: ScientificCalculator) -> None:
        """Test percentage calculation."""
        result = calculator.run("What is 15% of 200?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert result.result == 30.0

    def test_average(self, calculator: ScientificCalculator) -> None:
        """Test average calculation."""
        result = calculator.run("What is the average of 10, 20, and 30?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert result.result == 20.0


class TestLogarithms:
    """Test logarithmic functions."""

    def test_natural_log(self, calculator: ScientificCalculator) -> None:
        """Test natural logarithm."""
        result = calculator.run("What is the natural logarithm of e?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert abs(result.result - 1.0) < 0.0001

    def test_log_base_10(self, calculator: ScientificCalculator) -> None:
        """Test base-10 logarithm."""
        result = calculator.run("What is log base 10 of 100?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert abs(result.result - 2.0) < 0.0001


class TestConstants:
    """Test mathematical constants."""

    def test_get_pi(self, calculator: ScientificCalculator) -> None:
        """Test getting pi."""
        result = calculator.run("What is the value of pi?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert abs(result.result - 3.14159) < 0.001

    def test_get_e(self, calculator: ScientificCalculator) -> None:
        """Test getting Euler's number."""
        result = calculator.run("What is Euler's number e?")
        assert result.success, f"Failed: {result.error}, plan: {result.plan}"
        assert abs(result.result - 2.71828) < 0.001

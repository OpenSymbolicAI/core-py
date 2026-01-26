"""Scientific calculator agent implementation with decompositions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from opensymbolicai.blueprints import PlanExecute
from opensymbolicai.core import decomposition, primitive
from opensymbolicai.exceptions import PreconditionError, ValidationError

if TYPE_CHECKING:
    from opensymbolicai.llm import LLM, LLMConfig


class ScientificCalculator(PlanExecute):
    """A scientific calculator agent with standard mathematical operations.

    Provides primitive operations for basic arithmetic, trigonometry,
    logarithms, powers, and other scientific calculator functions.

    Decompositions demonstrate how to compose primitives to solve
    complex problems.
    """

    def __init__(self, llm: LLM | LLMConfig | None = None) -> None:
        if llm is not None:
            super().__init__(
                llm=llm,
                name="ScientificCalculator",
                description="A scientific calculator that performs mathematical operations.",
            )
        self._memory: float = 0.0
        self._last_result: float = 0.0

    # ==================== Basic Arithmetic ====================

    @primitive(read_only=True)
    def add_numbers(self, first_number: float, second_number: float) -> float:
        """Add two numbers together.

        Args:
            first_number: The first number to add.
            second_number: The second number to add.

        Returns:
            The sum (first_number + second_number).
        """
        result = first_number + second_number
        self._last_result = result
        return result

    @primitive(read_only=True)
    def subtract_numbers(self, minuend: float, subtrahend: float) -> float:
        """Subtract the subtrahend from the minuend.

        Args:
            minuend: The number to subtract from.
            subtrahend: The number to subtract.

        Returns:
            The difference (minuend - subtrahend).
        """
        result = minuend - subtrahend
        self._last_result = result
        return result

    @primitive(read_only=True)
    def multiply_numbers(self, first_factor: float, second_factor: float) -> float:
        """Multiply two numbers together.

        Args:
            first_factor: The first number to multiply.
            second_factor: The second number to multiply.

        Returns:
            The product (first_factor * second_factor).
        """
        result = first_factor * second_factor
        self._last_result = result
        return result

    @primitive(read_only=True)
    def divide_numbers(self, dividend: float, divisor: float) -> float:
        """Divide the dividend by the divisor.

        Args:
            dividend: The number to be divided.
            divisor: The number to divide by.

        Returns:
            The quotient (dividend / divisor).

        Raises:
            ZeroDivisionError: If divisor is zero.
        """
        if divisor == 0:
            raise PreconditionError(
                "Cannot divide by zero",
                code="DIVISION_BY_ZERO",
                details={"dividend": dividend, "divisor": divisor},
            )
        result = dividend / divisor
        self._last_result = result
        return result

    # ==================== Powers and Roots ====================

    @primitive(read_only=True)
    def raise_to_power(self, base: float, exponent: float) -> float:
        """Raise a base number to a given exponent.

        Args:
            base: The base number.
            exponent: The power to raise the base to.

        Returns:
            The result of base^exponent.
        """
        result = math.pow(base, exponent)
        self._last_result = result
        return result

    @primitive(read_only=True)
    def square_root_of(self, number: float) -> float:
        """Calculate the square root of a number.

        Args:
            number: The number to find the square root of (must be non-negative).

        Returns:
            The square root of the number.

        Raises:
            ValueError: If number is negative.
        """
        if number < 0:
            raise ValidationError(
                "Cannot calculate square root of negative number",
                code="INVALID_INPUT",
                details={"received": number, "expected": "non-negative number"},
                field="number",
            )
        result = math.sqrt(number)
        self._last_result = result
        return result

    # ==================== Trigonometric Functions ====================

    @primitive(read_only=True)
    def sine(self, angle_in_radians: float) -> float:
        """Calculate the sine of an angle.

        Args:
            angle_in_radians: The angle in radians.

        Returns:
            The sine of the angle.
        """
        result = math.sin(angle_in_radians)
        self._last_result = result
        return result

    @primitive(read_only=True)
    def cosine(self, angle_in_radians: float) -> float:
        """Calculate the cosine of an angle.

        Args:
            angle_in_radians: The angle in radians.

        Returns:
            The cosine of the angle.
        """
        result = math.cos(angle_in_radians)
        self._last_result = result
        return result

    @primitive(read_only=True)
    def tangent(self, angle_in_radians: float) -> float:
        """Calculate the tangent of an angle.

        Args:
            angle_in_radians: The angle in radians.

        Returns:
            The tangent of the angle.
        """
        result = math.tan(angle_in_radians)
        self._last_result = result
        return result

    # ==================== Angle Conversion ====================

    @primitive(read_only=True)
    def convert_degrees_to_radians(self, angle_in_degrees: float) -> float:
        """Convert an angle from degrees to radians.

        Args:
            angle_in_degrees: The angle measured in degrees.

        Returns:
            The angle measured in radians.
        """
        result = math.radians(angle_in_degrees)
        self._last_result = result
        return result

    @primitive(read_only=True)
    def convert_radians_to_degrees(self, angle_in_radians: float) -> float:
        """Convert an angle from radians to degrees.

        Args:
            angle_in_radians: The angle measured in radians.

        Returns:
            The angle measured in degrees.
        """
        result = math.degrees(angle_in_radians)
        self._last_result = result
        return result

    # ==================== Logarithms and Exponentials ====================

    @primitive(read_only=True)
    def natural_logarithm(self, number: float) -> float:
        """Calculate the natural logarithm (base e) of a number.

        Args:
            number: The number to calculate the logarithm of (must be positive).

        Returns:
            The natural logarithm (ln) of the number.

        Raises:
            ValueError: If number is not positive.
        """
        if number <= 0:
            raise ValidationError(
                "Logarithm requires positive input",
                code="INVALID_INPUT",
                details={"received": number, "expected": "positive number"},
                field="number",
            )
        result = math.log(number)
        self._last_result = result
        return result

    @primitive(read_only=True)
    def logarithm_base_10(self, number: float) -> float:
        """Calculate the base-10 logarithm of a number.

        Args:
            number: The number to calculate the logarithm of (must be positive).

        Returns:
            The base-10 logarithm of the number.

        Raises:
            ValueError: If number is not positive.
        """
        if number <= 0:
            raise ValidationError(
                "Logarithm requires positive input",
                code="INVALID_INPUT",
                details={"received": number, "expected": "positive number"},
                field="number",
            )
        result = math.log10(number)
        self._last_result = result
        return result

    @primitive(read_only=True)
    def exponential_e_to_power(self, exponent: float) -> float:
        """Calculate e (Euler's number) raised to a power.

        Args:
            exponent: The power to raise e to.

        Returns:
            The result of e^exponent.
        """
        result = math.exp(exponent)
        self._last_result = result
        return result

    # ==================== Constants ====================

    @primitive(read_only=True)
    def get_pi(self) -> float:
        """Get the mathematical constant pi (approximately 3.14159).

        Returns:
            The value of pi (ratio of circumference to diameter of a circle).
        """
        return math.pi

    @primitive(read_only=True)
    def get_eulers_number(self) -> float:
        """Get Euler's number e (approximately 2.71828).

        Returns:
            The value of e (base of natural logarithms).
        """
        return math.e

    # ==================== Memory Operations ====================

    @primitive(read_only=False)
    def memory_store(self, value: float) -> float:
        """Store a value in memory, replacing any existing value.

        Args:
            value: The value to store in memory.

        Returns:
            The value that was stored.
        """
        self._memory = value
        return self._memory

    @primitive(read_only=False)
    def memory_recall(self) -> float:
        """Recall the current value stored in memory.

        Returns:
            The value currently stored in memory.
        """
        return self._memory

    @primitive(read_only=False)
    def memory_add(self, value: float) -> float:
        """Add a value to the current memory (M+).

        Args:
            value: The value to add to memory.

        Returns:
            The new memory value after addition.
        """
        self._memory += value
        return self._memory

    @primitive(read_only=False)
    def memory_subtract(self, value: float) -> float:
        """Subtract a value from the current memory (M-).

        Args:
            value: The value to subtract from memory.

        Returns:
            The new memory value after subtraction.
        """
        self._memory -= value
        return self._memory

    @primitive(read_only=False)
    def memory_clear(self) -> float:
        """Clear the memory (set to zero).

        Returns:
            The memory value after clearing (0.0).
        """
        self._memory = 0.0
        return self._memory

    # ==================== Decompositions ====================

    @decomposition(
        intent="What is sine of 90 degrees?",
        expanded_intent="First convert 90 degrees to radians, then calculate the sine of that angle",
    )
    def _sine_90(self) -> float:
        """Calculate the sine of 90 degrees."""
        angle_rad: float = self.convert_degrees_to_radians(angle_in_degrees=90)
        sin_90: float = self.sine(angle_in_radians=angle_rad)
        return sin_90

    @decomposition(
        intent="What is cosine of 45 degrees?",
        expanded_intent="First convert 45 degrees to radians, then calculate the cosine of that angle",
    )
    def _cosine_45(self) -> float:
        """Calculate the cosine of 45 degrees."""
        angle_rad: float = self.convert_degrees_to_radians(angle_in_degrees=45)
        cos_45: float = self.cosine(angle_in_radians=angle_rad)
        return cos_45

    @decomposition(
        intent="Calculate the area of a circle with radius 5",
        expanded_intent="Get pi, then multiply pi by the square of the radius (pi * r^2)",
    )
    def _circle_area(self) -> float:
        """Calculate the area of a circle with radius 5."""
        pi: float = self.get_pi()
        radius_squared: float = self.raise_to_power(base=5, exponent=2)
        area: float = self.multiply_numbers(
            first_factor=pi, second_factor=radius_squared
        )
        return area

    @decomposition(
        intent="Calculate the hypotenuse of a right triangle with sides 3 and 4",
        expanded_intent="Square both sides, add them together, then take the square root (Pythagorean theorem: c = sqrt(a^2 + b^2))",
    )
    def _hypotenuse(self) -> float:
        """Calculate the hypotenuse using the Pythagorean theorem."""
        a_squared: float = self.raise_to_power(base=3, exponent=2)
        b_squared: float = self.raise_to_power(base=4, exponent=2)
        sum_of_squares: float = self.add_numbers(
            first_number=a_squared, second_number=b_squared
        )
        hypotenuse: float = self.square_root_of(number=sum_of_squares)
        return hypotenuse

    @decomposition(
        intent="What is 15% of 200?",
        expanded_intent="Divide the percentage by 100 to get the decimal, then multiply by the value",
    )
    def _percentage(self) -> float:
        """Calculate 15% of 200."""
        decimal: float = self.divide_numbers(dividend=15, divisor=100)
        result: float = self.multiply_numbers(first_factor=decimal, second_factor=200)
        return result

    @decomposition(
        intent="Convert 180 degrees to radians and find tan of it",
        expanded_intent="First convert 180 degrees to radians, then calculate the tangent of that angle",
    )
    def _tan_180(self) -> float:
        """Calculate the tangent of 180 degrees."""
        angle_rad: float = self.convert_degrees_to_radians(angle_in_degrees=180)
        tan_180: float = self.tangent(angle_in_radians=angle_rad)
        return tan_180

    @decomposition(
        intent="Calculate compound interest: principal 1000, rate 5%, time 3 years",
        expanded_intent="Use the formula A = P * e^(rt): multiply rate by time, raise e to that power, then multiply by principal",
    )
    def _compound_interest(self) -> float:
        """Calculate continuous compound interest."""
        rate_decimal: float = self.divide_numbers(dividend=5, divisor=100)
        rate_times_time: float = self.multiply_numbers(
            first_factor=rate_decimal, second_factor=3
        )
        growth_factor: float = self.exponential_e_to_power(exponent=rate_times_time)
        final_amount: float = self.multiply_numbers(
            first_factor=1000, second_factor=growth_factor
        )
        return final_amount

    @decomposition(
        intent="What is the average of 10, 20, and 30?",
        expanded_intent="Add all numbers together, then divide by the count of numbers",
    )
    def _average(self) -> float:
        """Calculate the average of three numbers."""
        sum_1: float = self.add_numbers(first_number=10, second_number=20)
        total: float = self.add_numbers(first_number=sum_1, second_number=30)
        average: float = self.divide_numbers(dividend=total, divisor=3)
        return average

    @decomposition(
        intent="What is the sine of 30 degrees?",
        expanded_intent="First convert 30 degrees to radians, then calculate the sine",
    )
    def _sine_30(self) -> float:
        """Calculate the sine of 30 degrees."""
        angle_rad: float = self.convert_degrees_to_radians(angle_in_degrees=30)
        result: float = self.sine(angle_in_radians=angle_rad)
        return result

    @decomposition(
        intent="Calculate cosine of 60 degrees",
        expanded_intent="First convert 60 degrees to radians, then calculate the cosine",
    )
    def _cosine_60(self) -> float:
        """Calculate the cosine of 60 degrees."""
        angle_rad: float = self.convert_degrees_to_radians(angle_in_degrees=60)
        result: float = self.cosine(angle_in_radians=angle_rad)
        return result

    @decomposition(
        intent="What is the circumference of a circle with radius 10?",
        expanded_intent="Get pi, multiply by 2, then multiply by the radius (circumference = 2 * pi * r)",
    )
    def _circumference(self) -> float:
        """Calculate the circumference of a circle with radius 10."""
        pi: float = self.get_pi()
        two_pi: float = self.multiply_numbers(first_factor=2, second_factor=pi)
        circumference: float = self.multiply_numbers(
            first_factor=two_pi, second_factor=10
        )
        return circumference

# OpenSymbolicAI Demo
# Generates: ../assets/demo.gif

Output ../assets/demo.gif

Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 14
Set Width 900
Set Height 550
Set Padding 20
Set LetterSpacing 0
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms

Type "cd examples/calculator"
Enter
Sleep 200ms
Type "uv run python run_calculator.py qwen3:1.7b -v"
Enter
Sleep 2500ms
Type "What is sine of 45 degrees plus cosine of 30 degrees?"
Enter
Sleep 15s

# Pause to show the plan and result
Sleep 4s

Type "quit"
Enter
Sleep 500ms

import unittest
import traceback
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


class RichTestResult(unittest.TextTestResult):
    def startTest(self, test):
        """Override to prevent dot-based test progress from being printed."""
        super().startTest(test)

    def addSuccess(self, test):
        super().addSuccess(test)
        console.print(
            Panel(Text(f"✔ PASSED: {test}", style="bold green"), expand=False)
        )

    def addFailure(self, test, err):
        super().addFailure(test, err)
        formatted_error = "".join(traceback.format_exception(*err))
        console.print(
            Panel(
                Text(f"✘ FAILED: {test}\n\n{formatted_error}", style="bold red"),
                expand=False,
            )
        )

    def addError(self, test, err):
        super().addError(test, err)
        formatted_error = "".join(traceback.format_exception(*err))
        console.print(
            Panel(
                Text(f"⚠ ERROR: {test}\n\n{formatted_error}", style="bold yellow"),
                expand=False,
            )
        )

    def printErrors(self):
        """Override this method to prevent unittest from printing default error logs."""
        pass

    def printSummary(self):
        """Suppress summary output from unittest (e.g., 'Ran X tests' and final OK/FAILED messages)."""
        pass

    def printTestsResults(self):
        """Override test completion message (e.g., prevent 'FAILED (failures=1)')."""
        pass


class RichTestRunner(unittest.TextTestRunner):
    def __init__(self, stream=None, descriptions=False, verbosity=0):
        super().__init__(stream, descriptions, verbosity)

    def _makeResult(self):
        return RichTestResult(self.stream, self.descriptions, self.verbosity)


if __name__ == "__main__":
    unittest.main(testRunner=RichTestRunner(), exit=False)

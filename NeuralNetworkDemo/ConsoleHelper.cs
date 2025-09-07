namespace NeuralNetworkDemo;
internal static class ConsoleHelper
{
	internal static string? ReadConsole(params string[] messages)
	{
		foreach (var message in messages)
		{
			Console.WriteLine(message);
		}

		Console.Write(" > ");
		return Console.ReadLine();
	}

	internal static void ReadForced(
		string inputDemandMsg = "Please enter your input:",
		string failedToReadMsg = "Failed to read. Try again.",
		Func<string, bool>? inputSuccessPredicate = null)
	{
		while (true)
		{
			var input = ReadConsole(inputDemandMsg);
			if (input is not null && (inputSuccessPredicate is null || inputSuccessPredicate(input)))
				break;

			Console.WriteLine(failedToReadMsg);
		}
	}
}

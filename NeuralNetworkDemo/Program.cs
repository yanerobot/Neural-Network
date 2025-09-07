using NeuralNetworkDemo;
using NeuralNetworkDemo.Demos;

DemoBase[] demos = new DemoBase[] { new XorDemo() };

Console.WriteLine("Select demo:");
for (int i = 0; i < demos.Length; i++)
{
	Console.WriteLine($"{i + 1} - {demos[i].Name}");
}
int selection = 0;
while(true)
{
	string? input = ConsoleHelper.ReadConsole("Choose one of the options selected:");
	if (int.TryParse(input, out selection) && selection - 1 < demos.Length && selection - 1 >= 0)
	{
		break;
	}

	Console.WriteLine("Couldn't parse an option into a valid selection.");
}

demos[selection - 1].Run();

using KKNeuralNetwork;
using Luna.ConsoleProgressBar;

namespace NeuralNetworkDemo.Demos;
internal class XorDemo : DemoBase
{
	protected override string Path => System.IO.Path.Combine("cfg", "nn-xor.weights");
	public override string Name => "XOR Demo";

	internal override void Run()
	{
		var nn = new NeuralNetwork(2, CostFunction.CostType.MSE);

		nn.AddLayers<FullyConnectedLayer>(Activation.ActivationType.ReLU, 4, 4);
		nn.AddLayers<FullyConnectedLayer>(Activation.ActivationType.Linear, 1);
		nn.LoadWeights(Path);
		string? shouldTrain = ConsoleHelper.ReadConsole("Should train (default: no)? [y/n]:");
		if (shouldTrain is "y" or "Y")
		{
			Train(nn);
		}

		nn.SaveWeights(Path);
		Test(nn);
	}

	internal override void Train(NeuralNetwork nn)
	{
		Random rand = new Random();
		int dataSetSize = 3000;
		var pb = new ConsoleProgressBar();
		int epochCnt = 10000;
		for (int i = 0; i < epochCnt; i++)
		{
			TrainingData[] dataSet = new TrainingData[dataSetSize];

			for (int j = 0; j < dataSetSize; j++)
			{
				int x = (int)Math.Round(rand.NextDouble());
				int y = (int)Math.Round(rand.NextDouble());
				var input = new TrainingData(new double[] { x, y }, new double[] { x ^ y });
				dataSet[j] = input;
			}

			pb.Report((double)i / epochCnt);

			nn.Learn(dataSet, 0.001);
		}

		pb.Report(1);
		pb.Dispose();
		Console.WriteLine();
		Console.WriteLine("Loading finished!");
	}

	internal override void Test(NeuralNetwork nn)
	{
		int a = ReadBinary();
		int b = ReadBinary();
		double[] result = nn.Calculate(new double[] { a, b });
		Console.WriteLine(result[0].ToString("F10"));
	}

	static int ReadBinary()
	{
		int res = 0;
		ConsoleHelper.ReadForced(
			inputDemandMsg: "Enter 1 or 0:",
			failedToReadMsg: "Couldn't parse input into a valid binary. Only 1 or 0 are allowed!",
			inputSuccessPredicate: input => int.TryParse(input, out res) && (res == 0 || res == 1)
		);

		return res;
	}
}

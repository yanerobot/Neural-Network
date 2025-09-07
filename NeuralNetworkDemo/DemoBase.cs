using KKNeuralNetwork;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkDemo;
internal abstract class DemoBase
{
	internal abstract void Run();
	protected abstract string Path { get; }
	public abstract string Name { get; }
	internal abstract void Train(NeuralNetwork nn);
	internal abstract void Test(NeuralNetwork nn);
}

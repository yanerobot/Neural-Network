using KKNeuralNetwork.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;


namespace KKNeuralNetwork
{
	/// <summary>
	/// The NeuralNetwork class represents an implementation of an artificial neural network with the ability to create layers, 
	/// select activation functions, and use various loss functions. 
	/// It includes methods for training the network on individual examples, batches of data, and large datasets using mini-batches. 
	/// Support is provided for both forward propagation and backpropagation, as well as logging the training process. 
	/// There is also functionality to save and load network weights for future use.
	/// </summary>
	public class NeuralNetwork
	{
		List<Layer> layers;
		int inputNodesCount;
		ICostFunction costFunction;

		/// <summary>
		/// Use this variable to get some info about the learning progress
		/// </summary>
		public ComposedData ComposedData;

		/// <summary>
		/// Create neural network
		/// </summary>
		/// <param name="inputNodesCount">Number of input nodes</param>
		/// <param name="costFunctionType">Select cost function from CostFunction.CostType enum</param>
		public NeuralNetwork(int inputNodesCount, CostFunction.CostType costFunctionType = CostFunction.CostType.MSE)
		{
			this.inputNodesCount = inputNodesCount;

			layers = new List<Layer>();
			costFunction = CostFunction.GetCostFunction(costFunctionType);
			ComposedData = new ComposedData();
		}

		/// <summary>
		/// Function you call after you created an object of type NeuralNetwork. You can have combination of different layers and activation functions
		/// </summary>
		/// <param name="activationType">Select activation function from Activation.ActivationType enum</param>
		/// <param name="sizes">Chose consequtive numbers of nodes (N). Each number represents a new layer with N amount of nodes</param>
		public void AddLayers<T>(Activation.ActivationType activationType, params int[] sizes) where T: Layer
		{
			for (int i = 0; i < sizes.Length; i++)
			{
				T layer = (T)Activator.CreateInstance(typeof(T), GetNodesIn(), sizes[i], activationType)!;

				layers.Add(layer);
			}

			int GetNodesIn() => layers.Count == 0 ? inputNodesCount : layers.Last().nodesOut;
		}

		/// <summary>
		/// Calculates an output with preset parameters. Used for trained Neural Network to simply get an answer
		/// </summary>
		public double[] Calculate(double[] input)
		{
			foreach (var layer in layers)
			{
				var layerLearnData = layer.ForwardPass(input);
				input = layerLearnData.a;
			}

			return input;
		}

		#region Learn Methods
		/// <summary>
		/// Use this function to train neural network on a single InputData
		/// </summary>
		public void Learn(TrainingData input, double learnRate)
		{
			var learnData = ForwardPass(input);
			BackPass(learnData, input);

			foreach (var layer in layers)
				layer.ApplyChanges(learnRate);
		}

		/// <summary>
		/// Use this function to train neural network on a single batch of InputData. Batch can be of any size. <br /><br />
		/// <b>NOTE:</b> Every item of a batch is trained independently with average adjustments. 
		/// </summary>
		/// <param name="batch">Array of a TraningData</param>
		/// <param name="learnRate">Learn Rate</param>
		public void Learn(TrainingData[] batch, double learnRate)
		{
			Parallel.For(0, batch.Length, (i) =>
			{
				var learnData = ForwardPass(batch[i]);
				BackPass(learnData, batch[i]);
			});

			foreach (var layer in layers)
				layer.ApplyChanges(learnRate / batch.Length);
		}

		/// <summary>
		/// Use this function to train neural network on a big dataset. Items in dataset will be split into batches of given size and for each batch "Learn" function will be called.
		/// </summary>
		/// <param name="dataset">Array of a TraningData</param>
		/// <param name="batchSize">Size of batch that dataSet will be split into</param>
		/// <param name="learnRate">The learning rate, which controls how much to adjust the model's weights with respect to the loss gradient.</param>
		public void Learn(TrainingData[] dataset, int batchSize, double learnRate)
		{
			for (int i = 0; i < dataset.Length / batchSize + Math.Sign(dataset.Length % batchSize); i++) // doing 1 additional iteration if (batch.Length % batchSize) != 0
			{
				var miniBatch = new TrainingData[Math.Min(dataset.Length - i * batchSize, batchSize)]; // getting the remainder if batchSize is bigger than the remainder

				for (int j = 0; j < miniBatch.Length; j++)
					miniBatch[j] = dataset[i * batchSize + j];

				Learn(miniBatch, learnRate);
			}
		}
		#endregion

		#region Save/Load
		/// <summary>
		/// Saves neural network state into the file at the path.
		/// </summary>
		/// <param name="path">Path to the file</param>
		public void SaveWeights(string path)
		{
			try
			{
				if (!File.Exists(path))
				{
					var dir = Path.GetDirectoryName(path);
					if (!Directory.Exists(dir))
					{
						Directory.CreateDirectory(dir);
					}

					var file = File.Create(path);
					file.Close();
					Console.WriteLine("File wasn't found at " + path + "\nCreating new instance.");
				}

				List<string> saveData = new List<string>();
				string networkId = "I_" + layers.Count + "_" + inputNodesCount; // simple id of the network
				for (int l = 0; l < layers.Count; l++)
				{
					networkId += "_" + layers[l].nodesOut + "_" + layers[l].nodesIn;
				}
				saveData.Add(networkId);
				// example: I_3_1_8_1_8_8_1_8

				for (int l = 0; l < layers.Count; l++)
				{
					for (int n = 0; n < layers[l].nodesOut; n++)
					{
						saveData.Add("b_" + l + "_" + n + "_" + layers[l].b[n] + "_" + layers[l].bVelocities[n]); // Bias for every node
						for (int j = 0; j < layers[l].nodesIn; j++)
						{
							saveData.Add(l + "_" + n + "_" + j + "_" + layers[l].w[n, j] + "_" + layers[l].wVelocities[n, j]); // Weight for every node in
						}
					}
				}
				File.WriteAllLines(path, saveData);
				Console.WriteLine("------ Successfully Saved ------");
			}
			catch (IOException ioEx)
			{
				Console.WriteLine("Saving failed: I/O error occurred while saving weights: " + ioEx.Message);
			}
			catch (Exception ex)
			{
				Console.WriteLine("Saving failed: Unexpected error occurred while saving weights: " + ex.Message);
			}
		}

		/// <summary>
		/// Loads saved neural network state from file at path
		/// </summary>
		/// <param name="path">Path to the file</param>
		/// <returns>True if data was sucessfully loaded. False otherwise</returns>
		public bool LoadWeights(string path)
		{
			try
			{
				if (!File.Exists(path))
				{
					Console.WriteLine("Loading failed: File doesn't exist at path " + path);
					return false;
				}

				var loadData = File.ReadAllLines(path);

				foreach (string line in loadData)
				{
					int l, i, j;
					var res = line.Split('_');

					if (res[0] == "I")
					{
						if (inputNodesCount != int.Parse(res[2]) || layers.Count != int.Parse(res[1]))
							return false;

						int charIndex = 2;
						for (int k = 0; k < layers.Count; k++)
						{
							charIndex++;
							if (layers[k].nodesOut != int.Parse(res[charIndex]))
								return false;
							charIndex++;
							if (layers[k].nodesIn != int.Parse(res[charIndex]))
								return false;
						}
						continue;
					}
					if (res[0] == "b")
					{
						l = int.Parse(res[1]);
						i = int.Parse(res[2]);
						layers[l].b[i] = double.Parse(res[3]);
						layers[l].bVelocities[i] = double.Parse(res[4]);
						continue;
					}

					l = int.Parse(res[0]);
					i = int.Parse(res[1]);
					j = int.Parse(res[2]);

					layers[l].w[i, j] = double.Parse(res[3]);
					layers[l].wVelocities[i, j] = double.Parse(res[4]);
				}
				Console.WriteLine("------ Successfully Loaded ------");
				return true;
			}
			catch (IOException ioEx)
			{
				Console.WriteLine("Loading failed: I/O error occurred while loading weights: " + ioEx.Message);
				return false;
			}
			catch (FormatException formatEx)
			{
				Console.WriteLine("Loading failed: Data format error occurred while loading weights: " + formatEx.Message);
				return false;
			}
			catch (Exception ex)
			{
				Console.WriteLine("Loading failed: Unexpected error occurred while loading weights: " + ex.Message);
				return false;
			}
		}

		#endregion

		#region Private

		/// <summary>
		/// Performs forward pass operation
		/// </summary>
		/// <returns>A list calculated data for every layer</returns>
		List<LayerLearnData> ForwardPass(TrainingData inputData)
		{
			if (inputData.data.Length != inputNodesCount || inputData.expected.Length != layers.Last().nodesOut)
				throw new ArgumentOutOfRangeException("Input data size mismatch");

			List<LayerLearnData> layerLearnDatas = new List<LayerLearnData>();

			double[] initialInputs = inputData.data;

			foreach (var layer in layers)
			{
				var layerLearnData = layer.ForwardPass(inputData.data);
				layerLearnDatas.Add(layerLearnData);
				inputData.data = layerLearnData.a;
			}

			var cost = costFunction.CalcCost(layerLearnDatas.Last().a, inputData.expected);
			ComposedData.UpdateData(ref cost, ref initialInputs, ref inputData.expected, ref layerLearnDatas.Last().a);
			//LogLearningState(cost, layerLearnDatas.Last().a[0], inputData.expected[0], initialInputs);
			return layerLearnDatas;
		}

		/// <summary>
		/// Uses the layer data from forward pass to perform backwards propagation and cache adjustments
		/// </summary>
		/// <param name="layerLearnDatas">Result of the ForwardPass</param>
		/// <param name="inputData">Input item that was used in respective ForwardPass method</param>
		void BackPass(List<LayerLearnData> layerLearnDatas, TrainingData inputData)
		{
			int last = layers.Count - 1;

			CalculateDerivMemoOutput(layers[last], layerLearnDatas[last], inputData.expected);
			CacheAdjustments(layers[last], layerLearnDatas[last].derivMemo, layerLearnDatas[last - 1].a);

			// Left and Right layers stand for the normal visualisation of Neural Network, where Input layer is leftmost and Output layer is rightmost
			// So, "left layer" means left neighbour of the current layer
			double[] leftActivations;
			for (int i = last - 1; i >= 0; i--)
			{
				CalculateDerivMemoHidden(layers[i], layerLearnDatas[i], layers[i + 1].w, layerLearnDatas[i + 1].derivMemo);

				if (i > 0) leftActivations = layerLearnDatas[i - 1].a;
				else leftActivations = inputData.data;

				CacheAdjustments(layers[i], layerLearnDatas[i].derivMemo, leftActivations);
			}
		}

		// Calcuating the derivatives of the Cost function to the Inputs. Finding local minima using gradient descend. All "dy/dx" stand for partial derivatives.

		/// <summary>
		/// Calculate derivative of the Output layer. 
		/// </summary>
		void CalculateDerivMemoOutput(Layer outputLayer, LayerLearnData outputLayerLearnData, double[] expected)
		{
			for (int i = 0; i < outputLayer.nodesOut; i++)
			{
				double dCdA = costFunction.CalcDerivative(outputLayerLearnData.a, expected, i);
				double dAdZ = outputLayer.activation.Derivative(outputLayerLearnData.z, i);

				/* CrossEntropy + Softmax
                 var result = layers[last].a[i];
                 if (expected[i] == 1)
                    result -= 1;
                */

				outputLayerLearnData.derivMemo[i] = dCdA * dAdZ; // dCdZ = dCdA * dAdZ
			}
		}

		// Calculating the derivative of a single Hidden layer by using dynamic programming technique of memoization

		/// <summary>
		/// Calculate derivative of a Hidden layer.
		/// </summary>
		void CalculateDerivMemoHidden(Layer curLayer, LayerLearnData curLayerLearnData, double[,] rightLayerWeights, double[] rightLayerDerivMemo)
		{
			for (int i = 0; i < curLayer.nodesOut; i++)
			{
				double dCdA = 0;
				for (int j = 0; j < rightLayerWeights.GetLength(0); j++) // rightLayerWeights.GetLength(0) = rightLayer.nodesOut (amount of nodes in NEXT layer)
				{
					dCdA += rightLayerWeights[j, i] * rightLayerDerivMemo[j];
				}
				double dAdZ = curLayer.activation.Derivative(curLayerLearnData.z, i); // dA/dZ

				curLayerLearnData.derivMemo[i] = dCdA * dAdZ;
			}
		}


		/// <summary>
		/// Perform calculted adjustments. <br />
		/// <b>NOTE:</b> This method only caches the adjustments. Call Layer.AdjustParameters to apply the changes.
		/// </summary>
		void CacheAdjustments(Layer layer, double[] derivMemo, double[] leftActivations)
		{
			lock (layer.adjustW)
				for (int i = 0; i < layer.nodesOut; i++)
					for (int j = 0; j < layer.nodesIn; j++)
						layer.adjustW[i, j] += derivMemo[i] * leftActivations[j];

			lock (layer.adjustB)
				for (int i = 0; i < layer.nodesOut; i++)
					layer.adjustB[i] += derivMemo[i];
		}

		#endregion
	}
}

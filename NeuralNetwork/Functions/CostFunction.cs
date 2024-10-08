﻿using System;


namespace KKNeuralNetwork
{
	/// <summary>
	/// Provides various cost functions for evaluating the difference between predicted
	/// output and the expected output in machine learning models. The cost functions are
	/// implemented as readonly structs for efficient performance.
	/// </summary>
	public readonly struct CostFunction
	{
		/// <summary>
		/// Defines the type of cost/loss function used to evaluate the performance of a model.
		/// Cost functions measure the difference between predicted and actual values, 
		/// guiding the optimization process during model training.
		/// </summary>
		public enum CostType
		{
			/// <summary>
			/// Mean Squared Error (MSE)
			/// Measures the average of the squares of the errors between predicted and actual values.
			/// Commonly used in regression tasks.
			/// </summary>
			MSE,

			/// <summary>
			/// Cross-Entropy
			/// Used primarily in classification tasks.
			/// It measures the difference between two probability distributions (predicted and true).
			/// </summary>
			CrossEntropy,

			/// <summary>
			/// Mean Squared Logarithmic Error (MSLE)
			/// Similar to MSE, but takes the logarithm of the predicted and actual values, useful when you care more about relative differences.
			/// </summary>
			MSLE,

			/// <summary>
			/// Mean Absolute Percentage Error (MAPE)
			/// Measures the percentage error between predicted and actual values.
			/// Often used in regression problems for percentage-based evaluations.
			/// </summary>
			MAPE
		}


		/// <summary>
		/// Returns the corresponding cost function implementation based on the specified cost type.
		/// </summary>
		/// <param name="costType">The type of cost function to return.</param>
		/// <returns>An instance of a class implementing ICostFunction.</returns>
		public static ICostFunction GetCostFunction(CostType costType)
		{
			switch (costType)
			{
				case CostType.MSE:
					return new MSE();
				case CostType.CrossEntropy:
					return new CrossEntropy();
				case CostType.MSLE:
					return new MSLE();
				case CostType.MAPE:
					return new MAPE();
				default:
					throw new ArgumentException();
			}
		}

		internal readonly struct MAPE : ICostFunction
		{
			public double CalcCost(double[] output, double[] expected)
			{
				double error = 0;
				for (int i = 0; i < output.Length; i++)
				{
					error += Math.Abs((expected[i] - output[i]) / (expected[i] + double.Epsilon));
				}
				return error / output.Length;
			}

			public double CalcDerivative(double[] output, double[] expected, int index)
			{
				var a = output[index];
				var e = expected[index];
				return (a / (e * e) - 1 / e) / CalcCost(output, expected);
			}
		}


		internal readonly struct MSLE : ICostFunction
		{
			public double CalcCost(double[] output, double[] expected)
			{
				double error = 0;
				for (int i = 0; i < output.Length; i++)
				{
					error += Math.Pow(Math.Log(output[i] + 1) - Math.Log(expected[i] + 1), 2);
				}
				return error;
			}

			public double CalcDerivative(double[] output, double[] expected, int index)
			{
				return 2 * (Math.Log(output[index] + 1) - Math.Log(expected[index] + 1)) / (output[index] + 1);
			}
		}

		internal readonly struct MSE : ICostFunction
		{
			public double CalcCost(double[] output, double[] expected)
			{
				double error = 0;
				for (int i = 0; i < output.Length; i++)
				{
					error += Math.Pow(output[i] - expected[i], 2);
				}
				return error / output.Length;
			}

			public double CalcDerivative(double[] output, double[] expected, int index)
			{
				return 2 * (output[index] - expected[index]) / output.Length;
			}
		}

		internal readonly struct CrossEntropy : ICostFunction
		{
			public double CalcCost(double[] output, double[] expected)
			{
				for (int i = 0; i < expected.Length; i++)
				{
					if (expected[i] == 1)
					{
						return -Math.Log(output[i]);
					}
				}
				return 1;
			}

			public double CalcDerivative(double[] output, double[] expected, int index)
			{
				if (expected[index] == 0)
					return 1;
				return -1 / output[index];
			}
		}
	}
}
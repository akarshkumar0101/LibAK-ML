package machinelearning.neuralnet;

import java.util.ArrayList;
import java.util.List;

import data.tuple.Tuple2D;
import math.AKMath;

//D = Data, A = Answer
public interface NeuralNetworkTrainer<D, A> {

	public abstract double[] rawToInputLayer(D d);

	public abstract double[] rawToOutputLayer(A a);

	/**
	 * @return a [? instance of D, ? instance of A];
	 */
	public abstract Tuple2D<D, A> getRandomRawData();

	// return a double such that arr[0] is the input and arr[1] is the
	// expectedOutput
	public default Tuple2D<double[], double[]> getRandomTrainingExample() {
		Tuple2D<D, A> da = this.getRandomRawData();
		return new Tuple2D<>(this.rawToInputLayer(da.getA()), this.rawToOutputLayer(da.getB()));
	}

	public default List<Tuple2D<double[], double[]>> getRandomTrainingExamples(int batchSize) {
		List<Tuple2D<double[], double[]>> examples = new ArrayList<>(batchSize);
		for (int i = 0; i < batchSize; i++) {
			examples.add(this.getRandomTrainingExample());
		}
		return examples;
	}

	public static final double STANDARD_ACCEPTABLE_ERROR = .05;

	public default double acceptableOutputError() {
		return NeuralNetworkTrainer.STANDARD_ACCEPTABLE_ERROR;
	}

	public default boolean isCorrect(double[] output, double[] expectedOutput) {
		for (int i = 0; i < output.length; i++) {
			if (Math.abs((output[i] - expectedOutput[i]) / expectedOutput[i]) > this.acceptableOutputError())
				return false;
		}
		return true;
	}

	public default double getCost(double[] output, double[] expectedOutput) {
		double cost = 0;
		for (int i = 0; i < output.length; i++) {
			cost += AKMath.sqr(output[i] - expectedOutput[i]);
		}
		return cost;
	}

	public default double getDerivativeOfCost(double output, double expectedOutput) {
		return 2 * (output - expectedOutput);
	}

	public abstract double getWeightLearningRate();

	public default double getBiasLearningRate() {
		return this.getWeightLearningRate();
	}

}

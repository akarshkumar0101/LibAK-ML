package machinelearning.neuralnet;

public interface InputSource {

	public abstract double getInput(int inputIndex);

	public static final InputSource EMPTY_INPUT_SOURCE = inputIndex -> 0.0;
}

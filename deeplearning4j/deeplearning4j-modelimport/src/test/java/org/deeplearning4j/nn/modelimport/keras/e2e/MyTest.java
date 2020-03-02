package org.deeplearning4j.nn.modelimport.keras.e2e;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.DisableOnDebug;
import org.junit.rules.TestRule;
import org.junit.rules.Timeout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.Assert;

import java.io.File;
import java.io.IOException;

/**
 * Created by susaneraly on 2/26/20.
 */
public class MyTest extends BaseDL4JTest {

    @Rule
    public TestRule timeout = new DisableOnDebug(new Timeout(300000));

    @Test
    public void runThis() throws InvalidKerasConfigurationException, IOException, UnsupportedKerasConfigurationException {
        String model_path = "/Users/susaneraly/SKYMIND/keras-test-import/src/main/resources/NER/NER_dl4j_functional_many_to_one.hdf5";
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights(model_path);
        System.out.println(model.summary());
        INDArray inputA = Nd4j.createFromNpyFile(new File("/Users/susaneraly/SKYMIND/keras-test-import/src/main/resources/NER/toy_in_1.npy"));
        INDArray inputB = Nd4j.createFromNpyFile(new File("/Users/susaneraly/SKYMIND/keras-test-import/src/main/resources/NER/toy_in_2.npy"));
        INDArray inputC = Nd4j.createFromNpyFile(new File("/Users/susaneraly/SKYMIND/keras-test-import/src/main/resources/NER/toy_in_3.npy"));
        INDArray expectedOut = Nd4j.createFromNpyFile(new File("/Users/susaneraly/SKYMIND/keras-test-import/src/main/resources/NER/toy_out.npy"));
        INDArray actual = model.output(new INDArray[] {inputA,inputB.reshape(1,32),inputC})[0];
        System.out.println("EXPECTED:\n" + expectedOut);
        System.out.println("ACTUAL:\n" + actual);
        Assert.isTrue(expectedOut.equalsWithEps(actual,1e-4));
    }
}

package b6;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.ArrayFeatureVector;
import org.openimaj.feature.ByteFV;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.VectorAggregator;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;


public class SIFT {
	DoGSIFTEngine engine;
	
	private String[] classes;
	private boolean trained = false;
	private ByteCentroidsResult clusters;
	private LiblinearAnnotator<FImage, String> ann;
	
	
	public SIFT(){
		engine = new DoGSIFTEngine();	
	}
	/*
	 * Makes some use of code snippets from http://www.openimaj.org/tutorial/classification101.html
	 * */
	public void trainImages(String[] classes, Map<String,FImage[]> images){
		this.classes = classes;
		GroupedDataset<String,ListDataset<FImage>,FImage> dataset = new MapBackedDataset<String,ListDataset<FImage>,FImage>();
		
	
		
		for(Entry<String,FImage[]> e : images.entrySet()){
			ListDataset<FImage> listdata = new ListBackedDataset<FImage>();
			for(FImage f : e.getValue()){
				listdata.add(f);
			}	
			dataset.put(e.getKey(), listdata);
		}
		
		Set<ByteFV> vectors = new HashSet<ByteFV>();

		
		List<LocalFeatureList<ByteDSIFTKeypoint>> featurevectors = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
		
		for(Entry<String,FImage[]> e : images.entrySet()){
			DenseSIFT sifter = new DenseSIFT(16,16);
			
			for(FImage f : e.getValue()){
				sifter.analyseImage(f);
				LocalFeatureList<ByteDSIFTKeypoint> featurePoints = sifter.getByteKeypoints();
			
				featurevectors.add(featurePoints);
			}
		}
		
		int means = 500;
		
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(means);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(featurevectors);
		clusters = km.cluster(datasource);
		HardAssigner<byte[], float[], IntFloatPair> assigner = clusters.defaultHardAssigner();
		
		DenseSIFT dsift = new DenseSIFT(5, 7);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
		FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);
		
		ann = new LiblinearAnnotator<FImage, String>(
	            extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(dataset);
		trained = true;
	}
	
	public ClassificationResult<String> classify(FImage image){
		if(trained){			
			return ann.classify(image);
		}
		else{
			return null;
		}
	}
	
	static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
	    PyramidDenseSIFT<FImage> pdsift;
	    HardAssigner<byte[], float[], IntFloatPair> assigner;

	    public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
	    {
	        this.pdsift = pdsift;
	        this.assigner = assigner;
	    }

	    public DoubleFV extractFeature(FImage object) {
	        FImage image = object.getImage();
	        pdsift.analyseImage(image);

	        BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

	        BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
	                bovw, 2, 2);

	        return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
	    }
	}
}

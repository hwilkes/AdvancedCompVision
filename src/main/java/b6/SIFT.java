package b6;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.openimaj.data.DataSource;
import org.openimaj.feature.ArrayFeatureVector;
import org.openimaj.feature.ByteFV;
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
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

public class SIFT {
	DoGSIFTEngine engine;
	
	private String[] classes;
	private boolean trained = false;
	private ByteCentroidsResult clusters;
	
	
	public SIFT(){
		engine = new DoGSIFTEngine();	
	}
	/*
	 * Makes some use of code snippets from http://www.openimaj.org/tutorial/classification101.html
	 * */
	public void trainImages(String[] classes, Map<String,FImage> images){
		this.classes = classes;
		
		Set<FImage> imageSet = new HashSet<FImage>();
		for(Entry<String,FImage> e : images.entrySet()){
			imageSet.add(e.getValue());
		}
		List<LocalFeatureList<Keypoint>> featurevectors = new ArrayList<LocalFeatureList<Keypoint>>();
		
		for(Entry<String,FImage> e : images.entrySet()){
			LocalFeatureList<Keypoint> keypoints =  engine.findFeatures(e.getValue());
			featurevectors.add(keypoints);
		}
		
		int means = 500;
		
		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(means);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<Keypoint, byte[]>(featurevectors);
		clusters = km.cluster(datasource);
		trained = true;
	}
	
	public String classify(FImage image){
		
		LocalFeatureList<Keypoint> imageKeypoints = engine.findFeatures(image);
		return null;
	}
	
	
}

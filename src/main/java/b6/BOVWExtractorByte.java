package b6;

import java.util.List;

import org.openimaj.feature.ByteFV;
import org.openimaj.feature.ByteFVComparison;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.AbstractDenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.local.engine.Engine;



public class BOVWExtractorByte<SIFT extends AbstractDenseSIFT<FImage>> implements BOVWExtractor<SparseIntFV> {

	List<ByteFV> dictionary;
	Engine<ByteDSIFTKeypoint,FImage> method;
	//SIFT method;
	
	public BOVWExtractorByte(List<ByteFV> dictionary, Engine<ByteDSIFTKeypoint,FImage> method)
	{
		this.dictionary = dictionary;
		this.method = method;
	}
	
	public SparseIntFV extractFeature(FImage object) {
		int[] wordCounts = new int[dictionary.size()];
		for(int i = 0; i < wordCounts.length; i++)
		{
			wordCounts[i] = 0;
		}
		
		//method.analyseImage(object);
		LocalFeatureList<ByteDSIFTKeypoint> keypoints  = method.findFeatures(object);
		for(ByteDSIFTKeypoint keypoint: keypoints)
		{
			ByteFV vector = keypoint.getFeatureVector();

			ByteFV nearestWord = null;
			double nearestDistance = Double.MAX_VALUE;
			
			for(ByteFV word : dictionary){
				double distance = ByteFVComparison.EUCLIDEAN.compare(word, vector);
				if(distance < nearestDistance){
					nearestDistance = distance;
					nearestWord = word;
				}
			}
			
			wordCounts[dictionary.indexOf(nearestWord)] ++;
			
		}
		
		return new SparseIntFV(wordCounts);
	}

}

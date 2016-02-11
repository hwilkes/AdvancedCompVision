package b6;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat4;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.FeatureDetector;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.analysis.pyramid.gaussian.GaussianOctave;
import org.openimaj.image.analysis.pyramid.gaussian.GaussianPyramid;
import org.openimaj.image.feature.local.descriptor.gradient.SIFTFeatureProvider;
import org.openimaj.image.feature.local.detector.dog.collector.Collector;
import org.openimaj.image.feature.local.detector.dog.collector.OctaveKeypointCollector;
import org.openimaj.image.feature.local.detector.dog.extractor.DominantOrientationExtractor;
import org.openimaj.image.feature.local.detector.dog.extractor.GradientFeatureExtractor;
import org.openimaj.image.feature.local.detector.dog.extractor.OrientationHistogramExtractor;
import org.openimaj.image.feature.local.detector.dog.pyramid.DoGOctaveExtremaFinder;
import org.openimaj.image.feature.local.detector.pyramid.BasicOctaveExtremaFinder;
import org.openimaj.image.feature.local.detector.pyramid.OctaveInterestPointFinder;
import org.openimaj.image.feature.local.engine.DoGSIFTEngineOptions;
import org.openimaj.image.feature.local.engine.Engine;
import org.openimaj.image.feature.local.keypoints.InterestPointKeypoint;
import org.openimaj.image.feature.local.keypoints.Keypoint;
//import org.openimaj.image.feature.local.keypoints.Keypoint;

public class SURFEngine implements Engine<Keypoint,FImage> {
	
	private FeatureDetector fd;
	
	public SURFEngine(){
		fd = FeatureDetector.create(FeatureDetector.SURF);
	}
	
	@Override
	public LocalFeatureList<Keypoint> findFeatures(FImage image) {
		Mat matimage = new Mat(image.pixels.length,image.pixels[0].length,CvType.CV_8UC1);
		for(int x = 0; x < image.pixels.length; x++){
			for(int y = 0; y < image.pixels[0].length; y++){
				matimage.put(x, y, image.pixels[x][y]);
			}
		}
		
		MatOfKeyPoint output = new MatOfKeyPoint();
		fd.detect(matimage, output);

		LocalFeatureList<Keypoint> toReturn = new MemoryLocalFeatureList<Keypoint>();
		org.opencv.features2d.KeyPoint[] matkeys = output.toArray();
		
		for(org.opencv.features2d.KeyPoint k : matkeys){
			Keypoint toAdd = new Keypoint();
			toAdd.x = (float) k.pt.x;
			toAdd.y = (float) k.pt.y;
			toAdd.scale = k.size;
			toAdd.ori = k.angle;
			toAdd.ivec = new byte[128];
			toReturn.add(toAdd);
		}
		
		return toReturn;
	}
	
}

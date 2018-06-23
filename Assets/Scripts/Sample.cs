using System.IO;
using UnityEngine;

using TensorFlow;
using System;

public class Sample : MonoBehaviour
{
    private const double MIN_SCORE_FOR_OBJECT_HIGHLIGHTING = 0.5;

    private TFGraph mGraph;
    private TFSession mSession;

    private void Start()
    {
        Initialize();
    }

    private void Initialize()
    {
        byte[] model = File.ReadAllBytes(Application.dataPath + "/Resources/frozen_inference_graph.pb");
        mGraph = new TFGraph();
        mGraph.Import(new TFBuffer(model));

        mSession = new TFSession(mGraph);

        TFTensor tensor = CreateTensorFromImageFile(Application.dataPath + "/Resources/input.jpg", TFDataType.UInt8);
        TFSession.Runner runner = mSession.GetRunner();

        runner
            .AddInput(mGraph["image_tensor"][0], tensor)
            .Fetch(
            mGraph["detection_boxes"][0],
            mGraph["detection_scores"][0],
            mGraph["detection_classes"][0],
            mGraph["num_detections"][0]);
        var output = runner.Run();

        var boxes = (float[,,])output[0].GetValue(jagged: false);
        var scores = (float[,])output[1].GetValue(jagged: false);
        var classes = (float[,])output[2].GetValue(jagged: false);
        var num = (float[])output[3].GetValue(jagged: false);

        DrawBoxes(boxes, scores, classes, MIN_SCORE_FOR_OBJECT_HIGHLIGHTING);
        Debug.LogError("Initialized");
    }

    private TFTensor CreateTensorFromImageFile(string file, TFDataType destinationDataType = TFDataType.Float)
    {
        var contents = File.ReadAllBytes(file);

        // DecodeJpeg uses a scalar String-valued tensor as input.
        var tensor = TFTensor.CreateString(contents);

        TFOutput input, output;

        // Construct a graph to normalize the image
        using (var graph = ConstructGraphToNormalizeImage(out input, out output, destinationDataType))
        {
            // Execute that graph to normalize this one image
            using (var session = new TFSession(graph))
            {
                var normalized = session.Run(
                    inputs: new[] { input },
                    inputValues: new[] { tensor },
                    outputs: new[] { output });

                return normalized[0];
            }
        }
    }

    private TFGraph ConstructGraphToNormalizeImage(out TFOutput input, out TFOutput output, TFDataType destinationDataType = TFDataType.Float)
    {
        const int W = 224;
        const int H = 224;
        const float Mean = 117;
        const float Scale = 1;

        var graph = new TFGraph();
        input = graph.Placeholder(TFDataType.String);

        output = graph.Cast(graph.Div(
            x: graph.Sub(
                x: graph.ResizeBilinear(
                    images: graph.ExpandDims(
                        input: graph.Cast(
                            graph.DecodeJpeg(contents: input, channels: 3), DstT: TFDataType.Float),
                        dim: graph.Const(0, "make_batch")),
                    size: graph.Const(new int[] { W, H }, "size")),
                y: graph.Const(Mean, "mean")),
            y: graph.Const(Scale, "scale")), destinationDataType);

        return graph;
    }

    private void DrawBoxes(float[,,] boxes, float[,] scores, float[,] classes, double minScore)
    {
        var x = boxes.GetLength(0);
        var y = boxes.GetLength(1);
        var z = boxes.GetLength(2);

        float ymin = 0, xmin = 0, ymax = 0, xmax = 0;

        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                if (scores[i, j] < minScore) continue;

                for (int k = 0; k < z; k++)
                {
                    var box = boxes[i, j, k];
                    switch (k)
                    {
                        case 0:
                            ymin = box;
                            break;
                        case 1:
                            xmin = box;
                            break;
                        case 2:
                            ymax = box;
                            break;
                        case 3:
                            xmax = box;
                            break;
                    }

                }

                int value = Convert.ToInt32(classes[i, j]);
                Debug.LogError("find " + value + " => " + scores[i, j]);
            }
        }
    }
}

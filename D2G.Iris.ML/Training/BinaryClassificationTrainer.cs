using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public class BinaryClassificationTrainer : IModelTrainer
    {
        private readonly MLContext _mlContext;
        private readonly TrainerFactory _trainerFactory;

        public BinaryClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
        {
            _mlContext = mlContext;
            _trainerFactory = trainerFactory;
        }

        public async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting binary classification using {config.TrainingParameters.Algorithm}...");

            var labelPipeline = _mlContext.Transforms.CopyColumns(
                    outputColumnName: "RawLabel", inputColumnName: config.TargetField)
                .Append(_mlContext.Transforms.Conversion.ConvertType(
                    outputColumnName: "Label", inputColumnName: "RawLabel", outputKind: DataKind.Boolean));
            var labeledData = labelPipeline.Fit(dataView).Transform(dataView);


            IDataView fixedData;
            if (labeledData.Schema.GetColumnOrNull("Features").HasValue)
            {

                var temp = labeledData.GetColumn<VBuffer<float>>("Features")
                    .Zip(labeledData.GetColumn<bool>("Label"), (feat, lbl) => new BinaryVector { Features = feat.GetValues().ToArray(), Label = lbl })
                    .ToList();
                var schemaDef = SchemaDefinition.Create(typeof(BinaryVector));
                schemaDef[nameof(BinaryVector.Features)].ColumnType =
                    new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);
                fixedData = _mlContext.Data.LoadFromEnumerable(temp, schemaDef);
            }
            else
            {
                fixedData = _mlContext.Transforms.Concatenate("Features", featureNames)
                    .Fit(labeledData)
                    .Transform(labeledData);
            }

            var split = _mlContext.Data.TrainTestSplit(
                fixedData,
                testFraction: config.TrainingParameters.TestFraction,
                seed: 42);

            var trainer = _trainerFactory.GetTrainer(
                config.ModelType,
                config.TrainingParameters);

            var pipeline = _mlContext.Transforms.NormalizeMinMax("Features")
                .AppendCacheCheckpoint(_mlContext)
                .Append(trainer)
                .Append(_mlContext.Transforms.CopyColumns("Probability", "Score"));


            var start = DateTime.Now;
            var model = await Task.Run(() => pipeline.Fit(split.TrainSet));
            Console.WriteLine($"Training completed in {(DateTime.Now - start).TotalSeconds:N1} sec");

            Console.WriteLine("Evaluating model...");
            var predictions = model.Transform(split.TestSet);
            var metrics = _mlContext.BinaryClassification.Evaluate(
                data: predictions,
                labelColumnName: "Label",
                scoreColumnName: "Score",
                predictedLabelColumnName: "PredictedLabel");
            ConsoleHelper.PrintBinaryClassificationMetrics(config.TrainingParameters.Algorithm, metrics);
            Console.WriteLine($"Confusion Matrix:\n{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");

            await ModelHelper.CreateModelInfo<BinaryClassificationMetrics, float>(
                metrics,
                fixedData,
                featureNames,
                config,
                processedData);

            var modelPath = $"BinaryClassification_{config.TrainingParameters.Algorithm}_Model.zip";
            mlContext.Model.Save(model, fixedData.Schema, modelPath);
            Console.WriteLine($"Model saved to: {modelPath}");

            return model;
        }

        private class BinaryVector
        {
            [VectorType]
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }
    }
}

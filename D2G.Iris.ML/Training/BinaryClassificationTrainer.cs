using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Training;
using D2G.Iris.ML.Utils;
using D2G.Iris.ML.Core.Interfaces;

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
            Console.WriteLine($"\nStarting binary classification model training using {config.TrainingParameters.Algorithm}...");

            try
            {
                var rows = mlContext.Data.CreateEnumerable<FeatureVector>(dataView, reuseRowObject: false)
                    .Select(row => new BinaryRow
                    {
                        Features = row.Features.Take(featureNames.Length).ToArray(), 
                        Label = row.Label > 0 
                    })
                    .ToList();

                var schema = SchemaDefinition.Create(typeof(BinaryRow));
                schema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);

                var typedData = mlContext.Data.LoadFromEnumerable(rows, schema);

                var dataSplit = mlContext.Data.TrainTestSplit(
                    typedData,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);

                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                var pipeline = mlContext.Transforms
                    .NormalizeMinMax("Features")
                    .Append(trainer)
                    .Append(mlContext.Transforms.CopyColumns("Probability", "Score"));

                var model = await Task.Run(() => pipeline.Fit(dataSplit.TrainSet));

                Console.WriteLine("Evaluating model...");
                var predictions = model.Transform(dataSplit.TestSet);
                var metrics = mlContext.BinaryClassification.Evaluate(
                    predictions,
                    labelColumnName: "Label",
                    scoreColumnName: "Score",
                    predictedLabelColumnName: "PredictedLabel");

                ConsoleHelper.PrintBinaryClassificationMetrics(config.TrainingParameters.Algorithm, metrics);
                Console.WriteLine($"Confusion Matrix:\n{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");

                await ModelHelper.CreateModelInfo<BinaryClassificationMetrics, float>(
                metrics,
                dataView,
                featureNames,
                config,
                processedData);

                var modelPath = $"BinaryClassification_{config.TrainingParameters.Algorithm}_Model.zip";
                mlContext.Model.Save(model, typedData.Schema, modelPath);
                Console.WriteLine($"Model saved to: {modelPath}");

                return model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during model training: {ex.Message}");
                throw;
            }
        }

        private class BinaryRow
        {
            [VectorType]
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }

        private class FeatureVector
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }
    }
}
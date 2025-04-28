using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Training;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public class BinaryClassificationTrainer
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
                // Convert to strongly-typed data with fixed schema
                var rows = mlContext.Data.CreateEnumerable<FeatureVector>(dataView, reuseRowObject: false)
                    .Select(row => new BinaryRow
                    {
                        Features = row.Features.Take(featureNames.Length).ToArray(), // Ensure fixed length
                        Label = row.Label > 0 // Convert to boolean for binary classification
                    })
                    .ToList();

                // Create schema with fixed vector size
                var schema = SchemaDefinition.Create(typeof(BinaryRow));
                schema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);

                // Load data with fixed schema
                var typedData = mlContext.Data.LoadFromEnumerable(rows, schema);

                // Split data
                var dataSplit = mlContext.Data.TrainTestSplit(
                    typedData,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);

                // Get trainer
                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                // Create training pipeline
                var pipeline = mlContext.Transforms
                    .NormalizeMinMax("Features")
                    .Append(trainer)
                    .Append(mlContext.Transforms.CopyColumns("Probability", "Score"));

                // Train model
                var model = await Task.Run(() => pipeline.Fit(dataSplit.TrainSet));

                // Evaluate model
                Console.WriteLine("Evaluating model...");
                var predictions = model.Transform(dataSplit.TestSet);
                var metrics = mlContext.BinaryClassification.Evaluate(
                    predictions,
                    labelColumnName: "Label",
                    scoreColumnName: "Score",
                    predictedLabelColumnName: "PredictedLabel");

                // Print metrics using ConsoleHelper
                ConsoleHelper.PrintBinaryClassificationMetrics(config.TrainingParameters.Algorithm, metrics);

                // Save model information using ModelHelper
                await ModelHelper.CreateModelInfo<BinaryClassificationMetrics, float>(
                metrics,
                dataView,
                featureNames,
                config,
                processedData);

                // Save model
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
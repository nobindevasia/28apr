using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Training;

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

                // Print metrics
                PrintMetrics(metrics);

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

        private void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine();
            Console.WriteLine("Model Evaluation Metrics:");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
            Console.WriteLine($"Area Under ROC Curve: {metrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"Area Under PR Curve: {metrics.AreaUnderPrecisionRecallCurve:F4}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F4}");
            Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:F4}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F4}");
            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F4}");
            Console.WriteLine();
            Console.WriteLine("Confusion Matrix:");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
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
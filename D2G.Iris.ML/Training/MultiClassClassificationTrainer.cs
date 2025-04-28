using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public class MultiClassClassificationTrainer
    {
        private readonly MLContext _mlContext;
        private readonly TrainerFactory _trainerFactory;

        public MultiClassClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
        {
            _mlContext = mlContext;
            _trainerFactory = trainerFactory;
        }

        private class ModelInput
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }

        public async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting multiclass classification model training using {config.TrainingParameters.Algorithm}...");
            //Console.WriteLine($"Total samples: {processedData.BalancedSampleCount:N0}");
            Console.WriteLine($"Selected features: {string.Join(", ", featureNames)}");

            try
            {
                // 1) Materialize so we can enforce the correct Features vector size
                Console.WriteLine("\nPreparing data for training...");
                var data = mlContext.Data
                    .CreateEnumerable<ModelInput>(dataView, reuseRowObject: false)
                    .Select(row => new ModelInput
                    {
                        Features = row.Features,
                        Label = row.Label
                    })
                    .ToList();

                // 2) Create a fixed schema so the Features vector has the right length
                var schema = SchemaDefinition.Create(typeof(ModelInput));
                schema["Features"].ColumnType =
                    new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);

                var fixedData = mlContext.Data.LoadFromEnumerable(data, schema);

                // 3) Split into train/test
                Console.WriteLine("\nSplitting data into training and test sets...");
                var splitData = mlContext.Data.TrainTestSplit(
                    fixedData,
                    testFraction: config.TrainingParameters.TestFraction,
                    seed: 42);

                Console.WriteLine($"Training set size: {splitData.TrainSet.GetRowCount():N0} rows");
                Console.WriteLine($"Test set size: {splitData.TestSet.GetRowCount():N0} rows");

                // 4) Build the pipeline
                // — First, normalize features
                // — Then replace the raw Int64 `Label` column *in place* with a Key<UInt32> version
                Console.WriteLine("\nConfiguring training pipeline...");
                IEstimator<ITransformer> pipeline = mlContext.Transforms
                    .NormalizeMinMax("Features")
                    // This maps your int64 labels into a key type *called* "Label",
                    // so the trainer's default labelColumnName="Label" now finds a Key<UInt32>.
                    .Append(mlContext.Transforms.Conversion
                        .MapValueToKey(outputColumnName: "Label", inputColumnName: "Label"))
                    .AppendCacheCheckpoint(mlContext);

                // 5) Append the trainer itself
                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                pipeline = pipeline
                    .Append(trainer)
                    // And finally map the predicted key back to the original value
                    .Append(mlContext.Transforms.Conversion
                        .MapKeyToValue("PredictedLabel", "PredictedLabel"));

                // 6) Train
                Console.WriteLine("\nTraining model...");
                var trainStartTime = DateTime.Now;
                var model = await Task.Run(() => pipeline.Fit(splitData.TrainSet));
                var trainingTime = DateTime.Now - trainStartTime;
                Console.WriteLine($"Training completed in {trainingTime.TotalSeconds:N1} seconds");

                // 7) Evaluate
                Console.WriteLine("\nEvaluating model...");
                // Now we can use the default labelColumnName="Label" since we replaced it.
                var predictions = model.Transform(splitData.TestSet);
                var metrics = mlContext.MulticlassClassification.Evaluate(
                    predictions);

                // Use ConsoleHelper to print metrics
                ConsoleHelper.PrintMultiClassClassificationMetrics(config.TrainingParameters.Algorithm, metrics);

                // 8) Save
                var modelPath = $"MultiClassClassification_{config.TrainingParameters.Algorithm}_Model.zip";
                mlContext.Model.Save(model, fixedData.Schema, modelPath);
                Console.WriteLine($"\nModel saved to: {modelPath}");

                return model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError during model training: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                if (ex.InnerException != null)
                    Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                throw;
            }
        }
    }
}
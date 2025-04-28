using Microsoft.ML;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using D2G.Iris.ML.Configuration;
using D2G.Iris.ML.Data;
using D2G.Iris.ML.Training;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            try
            {
                Console.WriteLine("Starting ML.NET Pipeline...");

                // Load configuration
                string configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "modelconfig.json");
                var configManager = new ConfigManager();
                var config = configManager.LoadConfiguration(configPath);

                // Setup SQL connection
                var sqlHandler = new SqlHandler(config.Database.TableName);
                sqlHandler.Connect(config.Database);

                // Get enabled fields
                var enabledFields = config.InputFields
                    .Where(f => f.IsEnabled)
                    .Select(f => f.Name)
                    .ToArray();

                var mlContext = new MLContext(seed: 42);

                Console.WriteLine("Loading data...");
                var dataLoader = new DatabaseDataLoader();
                var rawData = dataLoader.LoadDataFromSql(
                    sqlHandler.GetConnectionString(),
                    config.Database.TableName,
                    enabledFields,
                    config.ModelType,
                    config.TargetField,
                    config.Database.WhereClause);

                // Process the data
                Console.WriteLine("Processing data...");
                var dataProcessor = new DataProcessor();
                var processedData = await dataProcessor.ProcessData(
                    mlContext,
                    rawData,
                    enabledFields,
                    config,
                    sqlHandler); 

                // Train the model
                Console.WriteLine("Training model...");
                var modelTrainerFactory = new ModelTrainerFactory(mlContext);
                var modelTrainer = modelTrainerFactory.CreateTrainer(config.ModelType);

                await modelTrainer.TrainModel(
                    mlContext,
                    processedData.Data,
                    processedData.FeatureNames,
                    config,
                    processedData);

                Console.WriteLine("Pipeline completed successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in pipeline: {ex.Message}");
                Console.WriteLine($"Stack Trace: {ex.StackTrace}");
                if (ex.InnerException != null)
                {
                    Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                }
                throw;
            }
        }
    }
}
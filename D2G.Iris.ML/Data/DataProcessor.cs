using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.DataBalancing;
using D2G.Iris.ML.FeatureEngineering;

namespace D2G.Iris.ML.Data
{
    public class DataProcessor
    {
        public async Task<ProcessedData> ProcessData(
            MLContext mlContext,
            IDataView rawData,
            string[] enabledFields,
            ModelConfig config)
        {
            Console.WriteLine("\n=============== Processing Data ===============");

            IDataView processedData = rawData;
            string[] currentFeatures = enabledFields.Where(f => f != config.TargetField).ToArray();
            string selectionReport = string.Empty;

            // Original row count
            long originalCount = rawData.GetRowCount() ?? 0;
            long balancedCount = originalCount;

            // Create initial feature vector if needed
            if (!rawData.Schema.GetColumnOrNull("Features").HasValue)
            {
                var initialPipeline = mlContext.Transforms.Concatenate("Features", currentFeatures);
                processedData = initialPipeline.Fit(rawData).Transform(rawData);
            }

            // Determine execution order
            bool balancingFirst = config.DataBalancing.ExecutionOrder <= config.FeatureEngineering.ExecutionOrder;

            if (config.DataBalancing.Method != DataBalanceMethod.None &&
                config.FeatureEngineering.Method != FeatureSelectionMethod.None)
            {
                Console.WriteLine($"Processing order: {(balancingFirst ?
                    "Data Balancing then Feature Selection" :
                    "Feature Selection then Data Balancing")}");
            }

            try
            {
                if (balancingFirst)
                {
                    // Data Balancing First
                    if (config.DataBalancing.Method != DataBalanceMethod.None)
                    {
                        var balancer = new SmoteDataBalancer();
                        processedData = await balancer.BalanceDataset(
                            mlContext,
                            processedData,
                            currentFeatures,
                            config.DataBalancing,
                            config.TargetField);
                        balancedCount = processedData.GetRowCount() ?? originalCount;
                        Console.WriteLine($"Data balanced. New count: {balancedCount}");
                    }

                    // Then Feature Selection
                    if (config.FeatureEngineering.Method != FeatureSelectionMethod.None)
                    {
                        var selector = new CorrelationFeatureSelector(mlContext);
                        var result = await selector.SelectFeatures(
                            mlContext,
                            processedData,
                            currentFeatures,
                            config.ModelType,
                            config.TargetField,
                            config.FeatureEngineering);

                        processedData = result.transformedData;
                        currentFeatures = result.selectedFeatures;
                        selectionReport = result.report;
                        Console.WriteLine(selectionReport);
                    }
                }
                else
                {
                    // Feature Selection First
                    if (config.FeatureEngineering.Method != FeatureSelectionMethod.None)
                    {
                        var selector = new CorrelationFeatureSelector(mlContext);
                        var result = await selector.SelectFeatures(
                            mlContext,
                            processedData,
                            currentFeatures,
                            config.ModelType,
                            config.TargetField,
                            config.FeatureEngineering);

                        processedData = result.transformedData;
                        currentFeatures = result.selectedFeatures;
                        selectionReport = result.report;
                        Console.WriteLine(selectionReport);
                    }

                    // Then Data Balancing
                    if (config.DataBalancing.Method != DataBalanceMethod.None)
                    {
                        var balancer = new SmoteDataBalancer();
                        processedData = await balancer.BalanceDataset(
                            mlContext,
                            processedData,
                            currentFeatures,
                            config.DataBalancing,
                            config.TargetField);
                        balancedCount = processedData.GetRowCount() ?? originalCount;
                        Console.WriteLine($"Data balanced. New count: {balancedCount}");
                    }
                }

                return new ProcessedData
                {
                    Data = processedData,
                    FeatureNames = currentFeatures,
                    OriginalSampleCount = (int)originalCount,
                    BalancedSampleCount = (int)balancedCount,
                    FeatureSelectionReport = selectionReport,
                    FeatureSelectionMethod = config.FeatureEngineering.Method,
                    DataBalancingMethod = config.DataBalancing.Method,
                    DataBalancingExecutionOrder = config.DataBalancing.ExecutionOrder,
                    FeatureSelectionExecutionOrder = config.FeatureEngineering.ExecutionOrder
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during data processing: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                throw;
            }
        }
    }
}
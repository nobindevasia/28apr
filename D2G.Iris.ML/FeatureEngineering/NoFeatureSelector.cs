using System;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class NoFeatureSelector 
    {
        private readonly MLContext _mlContext;
        private readonly StringBuilder _report;

        public NoFeatureSelector(MLContext mlContext)
        {
            _mlContext = mlContext;
            _report = new StringBuilder();
        }

        public Task<(IDataView transformedData, string[] selectedFeatures, string report)> SelectFeatures(
            MLContext mlContext,
            IDataView data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            _report.Clear();
            _report.AppendLine("\nNo Feature Selection Applied");
            _report.AppendLine("----------------------------------------");
            _report.AppendLine($"Using all enabled features: {candidateFeatures.Length}");

            foreach (var feature in candidateFeatures)
            {
                _report.AppendLine($"- {feature}");
            }

            // Create pipeline to concatenate features
            var pipeline = mlContext.Transforms
                .Concatenate("Features", candidateFeatures);

            var transformedData = pipeline.Fit(data).Transform(data);

            return Task.FromResult((transformedData, candidateFeatures, _report.ToString()));
        }
    }
}
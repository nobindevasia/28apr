using System;
using System.Threading.Tasks;
using Microsoft.ML;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.DataBalancing
{
    public class NoDataBalancer
    {
        public Task<IDataView> BalanceDataset(
            MLContext mlContext,
            IDataView data,
            string[] featureNames,
            DataBalancingConfig config,
            string targetField)
        {
            Console.WriteLine("No data balancing applied - returning original dataset");
            return Task.FromResult(data);
        }
    }
}
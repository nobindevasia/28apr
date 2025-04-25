using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Interfaces
{
    public interface ISqlHandler
    {
        void Connect(DatabaseConfig dbConfig);
        string GetConnectionString();
    }
}
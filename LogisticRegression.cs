using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLibrary
{
    public class LogisticRegression
    {
        public double[] xVals { get; set; }
        public double[] yVals { get; set; }
        public double slope { get; set; }
        public double yIntercept { get; set; }
        public LogisticRegression(double[] _xVals, double[] _yVals)
        {
            xVals = _xVals;
            yVals = _yVals;
        }

        double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        (double, double) GradientDescent(double learningRate, double currSlope, double currInter)
        {
            double m_gradient = 0;
            double b_gradient = 0;
            int itemsNum = xVals.Length;
            for (int i = 0; i < itemsNum; i++)
            {
                double currPrediction = (xVals[i] * currSlope + currInter);
                m_gradient += 2 * xVals[i] * (Sigmoid(currPrediction) - yVals[i]) / itemsNum;
                b_gradient += 2 * (Sigmoid(currPrediction) - yVals[i]) / itemsNum;
            }

            double s = currSlope - m_gradient * learningRate;
            double b = currInter - b_gradient * learningRate;
            return (s, b);
        }

        public void fit(double learningRate, int epochs)
        {
            double ss = 1;
            double bb = 1;
            for (int e = 0; e <= epochs; e++)
            {
                (ss, bb) = GradientDescent(learningRate, ss, bb);
            }

            slope = ss;
            yIntercept = bb;
        }

        public bool predict(double x)
        {
            double pred = Sigmoid(x * slope + yIntercept);
            if(pred >= 0.5)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

    }
}

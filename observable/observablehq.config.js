/**
 * Observable Framework configuration for MinIVess dashboard.
 *
 * See: https://observablehq.com/framework/config
 */
export default {
  title: "MinIVess MLOps Dashboard",
  root: "src",
  output: "dist",
  theme: "dashboard",
  pages: [
    {
      name: "Model Performance",
      path: "/loss-comparison",
    },
    {
      name: "Training Curves",
      path: "/training-curves",
    },
    {
      name: "External Generalization",
      path: "/external-generalization",
    },
  ],
};

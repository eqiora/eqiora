// Container entries must use stable names rather than absolute build paths.
module.exports = {
  context: __dirname,
  optimization: { moduleIds: 'named' },
};

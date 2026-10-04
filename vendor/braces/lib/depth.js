'use strict';

// GHSA-vfj7-8cjw-p6xm: do not allow options to disable this safety bound.
module.exports = depth => {
  if (depth > 100) {
    throw new SyntaxError('Brace pattern nesting exceeds the maximum depth of 100');
  }
};

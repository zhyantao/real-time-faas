'use strict'

module.exports = async (event, context) => {
  const result = {
    'body': JSON.stringify(event.body),
    'content-type': event.headers["content-type"],
    'bilibili': '欢迎观看'
  }

  return context
    .status(200)
    .succeed(result)
}

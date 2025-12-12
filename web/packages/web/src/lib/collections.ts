import { createCollection } from "@tanstack/react-db"
import { electricCollectionOptions } from "@tanstack/electric-db-collection"
import {
  selectUsersSchema,
  selectChatThreadSchema,
  selectChatMessageSchema,
} from "@/db/schema"

export const usersCollection = createCollection(
  electricCollectionOptions({
    id: "users",
    shapeOptions: {
      url: new URL(
        "/api/users",
        typeof window !== "undefined"
          ? window.location.origin
          : "http://localhost:8712",
      ).toString(),
      parser: {
        timestamptz: (date: string) => new Date(date),
      },
    },
    schema: selectUsersSchema,
    getKey: (item) => item.id,
  }),
)

const baseUrl =
  typeof window !== "undefined"
    ? window.location.origin
    : "http://localhost:8712"

export const chatThreadsCollection = createCollection(
  electricCollectionOptions({
    id: "chat_threads",
    shapeOptions: {
      url: new URL("/api/chat-threads", baseUrl).toString(),
      parser: {
        timestamptz: (date: string) => new Date(date),
      },
    },
    schema: selectChatThreadSchema,
    getKey: (item) => item.id,
  }),
)

export const chatMessagesCollection = createCollection(
  electricCollectionOptions({
    id: "chat_messages",
    shapeOptions: {
      url: new URL("/api/chat-messages", baseUrl).toString(),
      parser: {
        timestamptz: (date: string) => new Date(date),
      },
    },
    schema: selectChatMessageSchema,
    getKey: (item) => item.id,
  }),
)

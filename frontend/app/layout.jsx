import "./globals.css";

export const metadata = {
  title: "Reply review",
  description: "Academy Agent review queue",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

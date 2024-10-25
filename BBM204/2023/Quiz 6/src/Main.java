import java.util.Scanner;

public class Main {

    public static String findPassword(int passwordLength, String encryptedMessage) {
        // Remove spaces from the encrypted message
        encryptedMessage = encryptedMessage.replace(" ", "");

        // Find the most frequent substring of the given length
        String password = "";
        int maxCount = 0;

        for (int i = 0; i <= encryptedMessage.length() - passwordLength; i++) {
            String substring = encryptedMessage.substring(i, i + passwordLength);
            int count = countSubstring(encryptedMessage, substring);

            if (count > maxCount) {
                maxCount = count;
                password = substring;
            }
        }

        return password;
    }

    public static int countSubstring(String str, String substring) {
        int count = 0;
        int index = 0;

        while ((index = str.indexOf(substring, index)) != -1) {
            count++;
            index += substring.length();
        }

        return count;
    }

    public static String cleanMessage(String encryptedMessage, String password) {

        while (encryptedMessage.contains(password)) {
            encryptedMessage = encryptedMessage.replace(password, "");
        }

        return encryptedMessage;
    }

    public static String decryptMessage(String encryptedMessage, String password) {
        StringBuilder decryptedMessage = new StringBuilder();
        int charnum = 0;

        for (int i = 0; i < encryptedMessage.length(); i++) {
            char ch = encryptedMessage.charAt(i);

            if (Character.isLetter(ch)) {
                // Calculate the shift amount based on the password

                int shift = Character.toLowerCase(password.charAt(charnum % password.length())) - 'a' + 1;
                char chx = Character.toLowerCase(ch);
                char decryptedChar = (char) (((chx + shift - 'a') % 26) + 'a');

                // System.out.println(
                // password.charAt(charnum % password.length()) + " -> " + shift + " -> " + chx
                // + " -> "
                // + decryptedChar);

                // Preserve the original case of the character
                if (Character.isUpperCase(ch)) {
                    decryptedChar = Character.toUpperCase(decryptedChar);
                }

                charnum++; // Increase the character number

                decryptedMessage.append(decryptedChar);
            }

            else {
                decryptedMessage.append(ch);
            }
        }

        return decryptedMessage.toString();
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Read the password length
        int passwordLength = scanner.nextInt();
        scanner.nextLine(); // Consume the remaining newline character

        // Read the encrypted message
        String encryptedMessage = scanner.nextLine();

        // Find the password
        String password = findPassword(passwordLength, encryptedMessage);

        // Clean the encrypted message
        String cleanedMessage = cleanMessage(encryptedMessage, password);

        // Print the cleaned message
        // System.out.println(cleanedMessage);

        // Decrypt the message
        String decryptedMessage = decryptMessage(cleanedMessage, password);

        // Print the password and decrypted message
        System.out.println(password);
        System.out.println(decryptedMessage);

        scanner.close();
    }
}
